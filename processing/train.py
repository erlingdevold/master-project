

from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
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





def calculate_classification_loss_and_accuracy(preds,targets,pos_weight=None):
    print(targets)

    preds = preds.view(targets.shape)
    loss = F.mse_loss(torch.log(preds+1).cpu().detach(),torch.log(targets).cpu().detach(),reduction='mean',)

    return loss.float()





class LitModel(pl.LightningModule):
    def __init__(self,classification=False,load_model=False,n_output=2,onehot=False,activation='sigmoid',threshold='_5'):
        super().__init__()

        self.threshold = threshold # just for hparams file
        self.onehot = onehot
        self.classification = classification
        self.load_model= load_model
        self.optimizer = None
        self.model = BERTModel(classification=classification,activation=activation,n_output=n_output)
        self.accuracy = torchmetrics.Accuracy(task = 'multiclass',num_classes=n_output)
        # self.loss = nn.MSELoss()

        self.save_hyperparameters()

        if self.load_model:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(f'lightning_logs/version_2/checkpoints/epoch=29-step=3630.ckpt',map_location=device,)
            self.model.load_state_dict(checkpoint,strict=False)
    
    def forward(self, batch,batch_idx,forward_type):
        src = batch['enc']
        dec = batch['dec']
        dec_mask = None
        src_mask = None
        mask_rate=.5
        r2 = 0
        if self.classification:
            targets = batch['target'].float()
            preds = self.model(src.float(),dec.float(),src_mask,dec_mask)
            if not self.onehot:
                loss = F.mse_loss(preds,targets)
                self.log(f'{forward_type}loss', loss, on_step=True, on_epoch=True, prog_bar=True)
                
                r2 = r2_score(targets.detach().cpu(),preds.detach().cpu())
                self.log(f'{forward_type}r2', r2, on_step=False, on_epoch=True, prog_bar=True)
                return loss,[],r2,preds
            else:
                loss = F.binary_cross_entropy(preds,targets)
                self.log(f'{forward_type}loss', loss, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                # preds = preds > .5
                acc = self.accuracy(preds.detach().cpu(),targets.detach().cpu())

                # acc = self.accuracy()
                self.log(f'{forward_type}acc', acc, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True,)
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
                          lr=1e-3,
                          weight_decay=1e-2,
                          amsgrad=True)
        return [optimizer], []

    def validation_step(self,batch,batch_idx ):
        loss, log,r2,preds = self.forward(batch,batch_idx,'val_')
        return loss
    
    def test_step(self,batch,batch_idx ):
        loss, log,r2,preds = self.forward(batch,batch_idx,'test_')
        return loss



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


    return train, test, val

def train_model(classification,regression_head_type : str = '',onehot=0,threshold='_5'):
    match regression_head_type:
        case 'synthetic ':
            selection = None

        case 'binary':
            selection = ["SAN","OTHER"]
            pass

        case 'multi':
            _, selection = find_max_number_species_code()
            selection = list(selection)
            print(f"training with species {selection}, number of species: {len(selection)}")

        case _ :
            pass

    train,test,val = split_train_test_val("ds/ds_labeled/")

    dataloader = create_dataloader if classification else create_synthetic_dataloader

    model = LitModel(classification=classification,n_output=len(selection),onehot=onehot,threshold=threshold)
    model.load_state_dict(torch.load("models/synthetic_model.ckpt")['state_dict'],strict=False)

    train_dl = dataloader(train,"ds/ds_labeled/segmented/","ds/labels_crimac_2021/",selection=selection,onehot=onehot,threshold=threshold)
    test_dl = dataloader(test,"ds/ds_labeled/segmented/","ds/labels_crimac_2021/",shuffle=False,selection=selection,onehot=onehot,threshold=threshold)
    val_dl = dataloader(val,"ds/ds_labeled/segmented/","ds/labels_crimac_2021/",shuffle=False,selection=selection,onehot=onehot,threshold=threshold)

    if torch.cuda.is_available():
        trainer = pl.Trainer(accelerator='gpu',devices=1,max_epochs=100)

    trainer.fit(model,train_dataloaders=train_dl,val_dataloaders=val_dl)
    trainer.save_checkpoint("models/model_classification.ckpt" if classification else "models/model_synthetic.ckpt")

    test_res = trainer.test(model,dataloaders=test_dl)

    print(test_res)


if __name__ == "__main__":

    
   # train_model(classification=False)

    train_model(classification=True,regression_head_type='multi',onehot=1)
