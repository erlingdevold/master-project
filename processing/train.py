

from typing import Any
import numpy as np
import torch
import time
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from loader import create_synthetic_dataloader

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from model import BERTModel




# def calculate_classification_loss_and_accuracy(preds,targets,pos_weight=None):
#     preds = preds.view(targets.shape)

#     if pos_weight:
#         loss = F.mse_loss(preds,targets,reduction='mean',pos_weight=pos_weight)
#     else:
#         loss = F.mse_loss(preds,targets,reduction='mean')
    
#     # Date weighting
#     preds = preds.detach().cpu().numpy()
#     return loss



def calculate_synthetic_loss(probs=None,
                             is_next_pred=None,
                             targets=None,
                             is_next=None,
                             prob_weight=None,
                             sum_loss=True):
    """
    Calculate loss
    """
    # Find loss for missing vectors
    is_next_preds = is_next_pred.view(is_next.shape)
    is_next_sequence = F.binary_cross_entropy_with_logits(is_next_preds,
                                                          is_next,
                                                          reduction='mean')

    is_next_acc_preds = torch.round(torch.sigmoid(is_next_pred))
    is_next_acc_preds = is_next_acc_preds.view(is_next.shape)
    is_next_acc = (is_next_acc_preds == is_next).sum().float()
    is_next_acc = is_next_acc / (is_next.shape[0])

    # Find loss for altered vectors
    altered_preds = probs.view(targets.shape)
    altered_vectors = F.binary_cross_entropy_with_logits(
        altered_preds, targets.cuda())

    altered_vectors_acc = torch.round(torch.sigmoid(probs))
    altered_vectors_acc = altered_vectors_acc.view(targets.shape)
    altered_vectors_acc = (altered_vectors_acc == targets).sum().float()
    altered_vectors_acc = altered_vectors_acc / \
        (targets.shape[0]*targets.shape[1])

    if not sum_loss:
        loss = ((prob_weight * altered_vectors) +
                ((1 - prob_weight) * is_next_sequence))
    else:
        loss = is_next_sequence + altered_vectors
    return loss, is_next_sequence.item(), altered_vectors.item(
    ), is_next_acc.item() * 100, altered_vectors_acc.item() * 100






class LitModel(pl.LightningModule):
    def __init__(self,):
        super().__init__()
        self.model = BERTModel()

        self.classification = False
        self.optimizer = None
        self.save_hyperparameters()
    
    def forward(self, batch,batch_idx,forward_type):
        src = batch['enc']
        dec = batch['dec']
        dec_mask = None
        src_mask = None
        mask_rate=.5
        if self.classification:
            targets = batch['target']
            preds = self.model(src,dec,src_mask,dec_mask)
            # loss,f1 = calculate_classification_loss_and_accuracy(preds,targets)
            loss = 0

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
            print(loss)
        self.log(f'{forward_type}loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{forward_type}is_next_axx', is_next_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{forward_type}altered_acc', altered_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss,[],targets,None

    def training_step(self, batch, batch_idx):
        loss, log,targets,preds = self.forward(batch,batch_idx,'train_')
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(params=self.parameters(),
                          lr=1e-3,
                          weight_decay=1e-2,
                          amsgrad=True)
        onecycle = OneCycleLR(
            optimizer=optimizer,
            max_lr=1e-3,
            epochs=1000,
            steps_per_epoch=1000,
            )  # Find suitable learning rate
        return [optimizer], [onecycle]

def get_n_targets(mask_rate, seq_length):
    # The number of targets should be an int
    n_targets = int(mask_rate * seq_length)
    return n_targets


def create_masked_LM_vectors(mask_rate, src, dec, wrng_seq):
    """
    Replaces vectors in the input tensor to make a task of determining which indexes have been changed.
    """
    # Do LM masking for each of the inputs
    src, src_trg = do_LM_masking(mask_rate, src, wrng_seq, dec)
    dec, dec_trg = do_LM_masking(mask_rate, dec, wrng_seq, src)

    # Concatenate the targets to create one big target vector
    masked_targets = torch.cat((src_trg, dec_trg), dim=1)

    return src.cuda(), dec.cuda(), masked_targets.cuda()


def do_LM_masking(mask_rate, tensor_to_mask, wrng_seq, src):
    number_of_targets = get_n_targets(mask_rate, tensor_to_mask.shape[1])
    # Make a target tensor
    targets = torch.zeros(tensor_to_mask.shape[0], tensor_to_mask.shape[1])
    #zero_vector = torch.zeros(tensor_to_mask.shape[2])

    # Go over input tensor
    for batch in range(tensor_to_mask.shape[0] - 1):
        # Sample n indexes
        indexes = random.sample(list(range(0, tensor_to_mask.shape[1] - 1)),
                                number_of_targets)
        for index in indexes:
            # A random number
            rand = random.random()

            # Select a random index from the wrong tensor
            random_batch_select = random.randint(0,
                                                 tensor_to_mask.shape[0] - 1)
            random_index_in_sequence = random.randint(
                0, tensor_to_mask.shape[1] - 1)

            # Change the vector to a vector from the src_sequence 80% of the time
            if rand <= 1.0:
                # Change the vector to the random vector 20% of the time
                mask_vector = wrng_seq[random_batch_select,
                                       random_index_in_sequence]
            else:
                # Change vector to vector from src 80% of the time
                mask_vector = src[batch, random_index_in_sequence]

            # Replace the tensor with our selected replacement tensor
            tensor_to_mask[batch, index] = mask_vector

            # Set this index to one in our target
            targets[batch, index] = 1.0
    return tensor_to_mask, targets

def randomize_next_sequence(dec, wrng_seq, prob=0.5):
    """
    With a given probability, a sequence will be exhanged for a sequence not actually following the previous sequence. The is_next variable is then also set to false. This creates the is_next sequence dataset.
    """
    is_next = torch.ones(dec.shape[0])
    for i in range(dec.shape[0]):
        if random.random() < prob:
            # Find random index
            # print(wrng_seq.shape[0])
            idx = random.randint(0, wrng_seq.shape[0] - 1)
            # Store the sequence from dec
            #tmp = dec[i]
            # Replace sequence of dec with a random one
            dec[i] = wrng_seq[idx]
            # Set the
            #wrng_seq[idx] = tmp
            is_next[i] = 0.0
    return dec, wrng_seq, is_next.cuda()

if __name__ == "__main__":

    # ds = ()
    # train_loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
    dl = create_synthetic_dataloader("ds/ds_labeled/segmented/","ds/labels_crimac_2021/")

    model = LitModel()
    if torch.cuda.is_available():
        trainer = pl.Trainer(accelerator='gpu',devices=1,)


    trainer.fit(model,train_dataloaders=dl)
    
    #trainer.test(model)

