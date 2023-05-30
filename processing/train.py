

from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
import numpy as np
import torch
import time
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from loader import create_synthetic_dataloader,create_dataloader
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from train_functions import get_n_targets,create_masked_LM_vectors,do_LM_masking,calculate_synthetic_loss, randomize_next_sequence
from sklearn.metrics import r2_score,accuracy_score
from model import BERTModel

import torchmetrics
from to_utc import find_max_number_species_code

from lightning.pytorch.callbacks.early_stopping import EarlyStopping



def calculate_classification_loss_and_accuracy(preds,targets,pos_weight=None):
    print(targets)

    preds = preds.view(targets.shape)
    loss = F.mse_loss(torch.log(preds+1).cpu().detach(),torch.log(targets).cpu().detach(),reduction='mean',)

    return loss.float()

def gaussian_function(x, mu,sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def calculate_loss_resampling_weight(dates : list = [],mu=0,sig=2000):

    weights = np.ones((len(dates),len(dates[0])))

    for i,target in enumerate(dates):
        sum_w = [1/2 + gaussian_function(np.array(target[i]),mu,sig)/2 for i in range(len(target))]

        for j, l in enumerate(sum_w):
            if len(l) == 0:
                sum_w[j] = 1
            else:
                sum_w[j] = sum(l)/len(l)
        weights[i] = np.array(sum_w)
    
    return weights

class LitModel(pl.LightningModule):
    def __init__(self,classification=False,load_model=False,n_output=2,onehot=False,activation='sigmoid',threshold='_5',criterion='bce',temporal=False,sig=500,steps_per_epoch=1000,lr = 1e-3, epoch=30):
        super().__init__()

        self.threshold = threshold # just for hparams file
        self.lr = lr
        self.epoch = epoch
        self.steps_per_epoch = steps_per_epoch
        self.onehot = onehot
        self.classification = classification
        self.temporal = temporal
        self.sig = sig
        self.load_model= load_model
        self.optimizer = None
        self.criterion = criterion
        self.n_output = n_output
        self.model = BERTModel(classification=classification,activation=activation ,n_output=n_output)
        if self.criterion == 'bce':
            self.accuracy = torchmetrics.Accuracy(task = 'binary',num_classes=n_output,threshold=0.5)
            self.precision = torchmetrics.Precision(task = 'binary',num_classes=n_output,threshold=0.5)
            self.recall = torchmetrics.Recall(task = 'binary',num_classes=n_output,threshold=0.5)
            self.mcc = torchmetrics.MatthewsCorrCoef(task = 'binary',num_classes=n_output,threshold=0.5)
        else:
            self.mae = torchmetrics.MeanAbsoluteError()
        # self.loss = nn.MSELoss()

        self.save_hyperparameters()

        if self.load_model:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(f'lightning_logs/version_2/checkpoints/epoch=29-step=3630.ckpt',map_location=device,)
            self.model.load_state_dict(checkpoint,strict=False)
    
    def forward(self, batch,batch_idx,forward_type,training=True):
        src = batch['enc']
        dec = batch['dec']
        dec_mask = None
        src_mask = None
        mask_rate=.5
        r2 = 0

        if self.classification:
            if self.temporal:
                weights = batch['date']
                weights = calculate_loss_resampling_weight(weights,sig=self.sig)
                weights = torch.tensor(weights)
            else:
                weights = torch.ones_like(batch['target'])
            
            weights = weights.to(self.device)

            targets = batch['target'].float()
            preds = self.model(src.float(),dec.float(),src_mask,dec_mask)
            if self.criterion == 'mse':
                loss = F.mse_loss(preds,targets,reduction='none')
                loss = (loss*weights).sum() / loss.sum()

                self.log(f'{forward_type}loss', loss, on_step=True, on_epoch=True, prog_bar=True)
                
                r2 = r2_score(targets.detach().cpu(),preds.detach().cpu())
                mae = self.mae(preds,targets).to(self.device)
                self.log(f'{forward_type}r2', r2, on_step=True, on_epoch=True, prog_bar=True)
                self.log(f'{forward_type}mae', mae.detach().cpu(), on_step=True, on_epoch=True, prog_bar=True)
                return loss,[],r2,preds
            else:
                # loss = F.binary_cross_entropy(preds,targets,weight=weights if self.temporal else None)
                loss = F.binary_cross_entropy_with_logits(preds,targets,weight=weights if self.temporal else None)

                self.log(f'{forward_type}loss', loss, on_step=True, on_epoch=True, prog_bar=True,)

                mcc = self.mcc(preds,targets).to(self.device)
                preds = preds.detach().cpu()
                targets = targets.detach().cpu()
                acc = self.accuracy(preds,targets)
                precision = self.precision(preds,targets)
                recall = self.recall(preds,targets)

                self.log(f'{forward_type}mcc', mcc.detach().cpu().float(), on_step=True, on_epoch=True, prog_bar=False,)
                self.log(f'{forward_type}acc', acc, on_step=True, on_epoch=True, prog_bar=True,)
                self.log(f'{forward_type}precision', precision, on_step=True, on_epoch=True, prog_bar=True,)
                self.log(f'{forward_type}recall', recall, on_step=True, on_epoch=True, prog_bar=True,)

                return loss,[],acc,preds 


        else:
            wrng_seq = batch['target']
            dec,wrong_seq, is_next = randomize_next_sequence(dec,wrng_seq)
            src,dec,targets = create_masked_LM_vectors(mask_rate,src,dec,wrong_seq)
            
            transformer_out,probs,is_next_pred = self.model(src,dec,src_mask,dec_mask)

            loss,next_seq,altered_loss,is_next_acc,altered_acc = calculate_synthetic_loss(
                probs=probs,
                is_next_pred=is_next_pred,
                targets=targets,
                is_next=is_next,
            )
            self.log(f'{forward_type}is_next_axx', is_next_acc, on_step=True, on_epoch=True, prog_bar=True)
            self.log(f'{forward_type}altered_acc', altered_acc, on_step=True, on_epoch=True, prog_bar=True)


        return loss,[],r2,None

    def training_step(self, batch, batch_idx):
        loss, log,r2,preds = self.forward(batch,batch_idx,'train_')
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(params=self.parameters(),
                          lr=self.lr,
                          weight_decay=1e-2,
                          amsgrad=True)
        onecycle = OneCycleLR(optimizer,max_lr=self.lr,epochs=self.epoch,steps_per_epoch=self.steps_per_epoch)
        return [optimizer], [onecycle]

    def validation_step(self,batch,batch_idx ):
        loss, log,r2,preds = self.forward(batch,batch_idx,'val_')
        return loss
    
    def test_step(self,batch,batch_idx ):
        loss, log,r2,preds = self.forward(batch,batch_idx,'test_')
        return loss

    def test_step_end(self,outputs):
        print(outputs)
        return outputs

import os
def split_train_test_val(dir,train_size=0.8,test_size=0.1,val_size=0.1):
    """
    Splits the data into train, test and validation sets.
    """
    # Get all the files
    files = os.listdir(dir)
    # Shuffle the files
    random.shuffle(files)
    # Split into train, test and validation
    train = files[:int(len(files) * train_size)]
    test = files[int(len(files) * train_size):int(len(files) * (train_size + test_size))]
    val = files[int(len(files) * (train_size + test_size)):]
    print("train,test,val")
    print(len(train),len(test),len(val))


    return train, test, val

def train_model(classification,regression_head_type : str = '',onehot=0,threshold='_5',criterion='bce',activation='sigmoid',temporal=False, bsz=8):
    match regression_head_type:
        case 'synthetic ':
            selection = None

        case 'binary':
            selection = ["SAN"]

        case 'multi':
            _, selection = find_max_number_species_code(T=threshold)
            selection = list(selection)
            print(f"training with species {selection}, number of species: {len(selection)}")

        case _ :
            pass

    train,test,val = split_train_test_val("ds/ds_labeled/")

    dataloader = create_dataloader if classification else create_synthetic_dataloader

    model = LitModel(classification=classification,n_output=len(selection),onehot=onehot,threshold=threshold,criterion=criterion,activation=activation,temporal=temporal, steps_per_epoch=len(train) // bsz,epoch=40,lr=1e-3)
    model.load_state_dict(torch.load("models/synthetic_model.ckpt")['state_dict'],strict=False)

    train_dl = dataloader(train,"ds/ds_labeled/segmented/","ds/labels_crimac_2021/",selection=selection,onehot=onehot,threshold=threshold,bsz=bsz)
    test_dl = dataloader(test,"ds/ds_labeled/segmented/","ds/labels_crimac_2021/",shuffle=False,selection=selection,onehot=onehot,threshold=threshold,bsz=bsz)
    val_dl = dataloader(val,"ds/ds_labeled/segmented/","ds/labels_crimac_2021/",shuffle=False,selection=selection,onehot=onehot,threshold=threshold,bsz=bsz)
    print(len(train_dl))
    print(len(test_dl))
    print(len(val_dl))

    if torch.cuda.is_available():
        trainer = pl.Trainer(accelerator='gpu',devices=1,max_epochs=40) #,callbacks=[EarlyStopping(monitor='val_loss_epoch',mode='min',patience=5)])

    trainer.fit(model,train_dataloaders=train_dl,val_dataloaders=val_dl)
    trainer.save_checkpoint(f"models/model_classification_{regression_head_type}{threshold}_{onehot}_{temporal}.ckpt" if classification else "models/model_synthetic.ckpt")

def test_model():
    model = LitModel(classification=True,n_output=90,onehot=1,threshold='_5')
    _,_,test = split_train_test_val("ds/ds_labeled/")
    model.load_state_dict(torch.load("lightning_logs/version_0/checkpoints/epoch=99-step=11800.ckpt")['state_dict'],strict=False)
    model.eval()
    model.freeze()
    test_dl = create_dataloader(test,"ds/ds_labeled/segmented/","ds/labels_crimac_2021/",shuffle=False)

    auroc = torchmetrics.AUROC(task='binary',num_classes=90)

    for i,batch in enumerate(test_dl):

        y_hat = model(batch,i)
        batch['target'] = batch['target'].cpu().numpy()

        auc = auroc(y_hat,batch['target'])

        print(auc)





import argparse


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--head_type', type=str, default='multi')
    argparser.add_argument('--onehot', type=int, default=1)
    argparser.add_argument('--threshold', type=str, default='10')
    argparser.add_argument('--temporal', type=int, default=1)
    
    args = argparser.parse_args()

    criterion = 'bce'
    activation = 'sigmoid'

    if not args.onehot:
        criterion = 'mse'
        activation = ''

    train_model(classification=True,regression_head_type=args.head_type,onehot=args.onehot,threshold="_" + args.threshold,criterion=criterion,activation=activation,temporal=args.temporal)
