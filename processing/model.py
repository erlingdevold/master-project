import torch
from torch import nn,Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset
import math


class BERTModel(nn.Module):
    def __init__(self,input_d:int = 526, d_model:int = 512, nhead: int=8, d_hid:int=2048,n_layers:int=6,dropout:float=.5,seq_length=256,classification=False,n_output : int = 2,activation=None,onehot=False):
        super().__init__()

        self.model_type = 'Transformer'
        self.activation = activation
        self.onehot = onehot
        self.classification = classification
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout,activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers,norm=nn.LayerNorm(d_model))
        self.upsample = nn.Linear(input_d,d_model,bias=False)

        if not self.classification:
            self.prob1 = nn.Linear(d_model,1)
            self.dropout1 = nn.Dropout(dropout)
            self.prob2 = nn.Linear(d_model*seq_length*2,1)
            self.dropout2= nn.Dropout(dropout)
        else:
            self.prob4 = nn.Linear(d_model*seq_length*2,n_output)
            self.dropout_4 = nn.Dropout(dropout)

    def forward(self,src:Tensor,dec,src_mask : Tensor,trg_mask:Tensor)-> Tensor:
        dec = dec+1
        src = torch.cat((src,dec),dim=1)

        src = self.upsample(src)
        src =self.pos_encoder(src)
        src = src.permute(1,0,2)

        encoded = self.transformer_encoder(src)

        if self.classification:
            return self.classification_output(encoded)
        else:
            return self.output(encoded)

    def output(self,transformer_out):
        probs = 0
        is_next = 0
        transformer_out = transformer_out.permute(1,0,2)
        recon_vec = transformer_out.reshape(
            transformer_out.shape[0],
            transformer_out.shape[1] * transformer_out.shape[2])

        altered_vecs = self.prob1(self.dropout1(torch.relu(transformer_out)))

        is_next = self.prob2(self.dropout2(torch.relu(recon_vec)))

        return transformer_out,altered_vecs,is_next
    def classification_output(self,transformer_out):
        transformer_out = transformer_out.permute(1,0,2)
        recon_vec = transformer_out.reshape(transformer_out.shape[0],-1)

        if self.activation == 'sigmoid':
            head = self.prob4(self.dropout_4(torch.relu(recon_vec)))
            head = torch.sigmoid(head)
        else:
            head = self.prob4(self.dropout_4(torch.relu(recon_vec)))

        return head

   

class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int, dropout: float = .1, max_len: int = 5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len,d_model)
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    
    def forward(self,x:Tensor)-> Tensor:
        x = x + Tensor(self.pe[:,:x.size(1)])
        return self.dropout(x)




if __name__ == "__main__":

    model = BERTModel()

    src = torch.rand(10,128,232)

    dec = torch.rand(10,128,232)

    src_mask = torch.rand(10,128,128)

    trg_mask = torch.rand(10,128,128)

    preds = model(src,dec,src_mask,trg_mask)

    




