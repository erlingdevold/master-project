

import numpy as np
import torch
import time
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import lightning.pytorch as pl




def calculate_classification_loss_and_accuracy(preds,targets,pos_weight=None):
    preds = preds.view(targets.shape)

    if pos_weight:
        loss = F.mse_loss(preds,targets,reduction='mean',pos_weight=pos_weight)
    else:
        loss = F.mse_loss(preds,targets,reduction='mean')
    
    # Date weighting
    preds = preds.detach().cpu().numpy()
    return loss







class LitModel(pl.LightningModule):
    def __init__(self,device='gpu',lr=1e-3,weight_decay=1e-3):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.epoch = 0
        self.batch = 0

        self.pos_weight = None

if __name__ == "__main__":


