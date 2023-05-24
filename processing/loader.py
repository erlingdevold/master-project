import copy
from torch.utils.data import DataLoader
import xarray as xr
import torch

from to_utc import create_delta_time
from normalizer import transform_sv, rotate_image
import os,json
import matplotlib.pyplot as plt

def read_dir(dir : str,extension : str = ""):

    dirs = os.listdir(dir)
    dirs = [x for x in dirs if x.endswith(".npy")]
    dirs.sort()

    return dirs

def load_json(path):
    """Load a json file."""
    with open(path, "r") as f:
        return json.load(f)

def load_npy(path):
    """Load a npy sv"""
    return np.load(path)
def read_file(file : str):
    
    return xr.open_zarr(file)

def read_meta(file :str):
    with open(file,"r") as f:
        return f.read()

import numpy as np

def transform_labels_json(annotation : dict,truth:str,selection :list = [],):
    """
    Exports labels selected from json to array,
    the selected labels are an index, rest is inserted on -1 (other species)
    """
    sz = len(selection)+1

    if len(selection) == 0:
        selection = list(annotation.keys())
        sz = len(selection)

    arr = np.empty(sz)

    date_arr = []

    for i, key in enumerate(annotation):
        if key not in selection: # other species
            arr[-1] += annotation[key]['weight'] 
            continue
        
        arr[i] = annotation[key]['weight']
        date_arr.append(create_delta_time(truth, annotation[key]['date']))
    
    return arr, date_arr,selection


import random

def cut_dataset( start_at, cut_at, dataset, cut_ends):
    """
    Cuts the dataset to fit our model vector size
    Hakon maloy
    """

    if cut_ends:
        dataset = dataset[:, start_at:cut_at]
    return dataset
class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self,example_dir, annotations_dir, transform=None,target_transform=None):
        self.example_dir_path = example_dir
        self.annotations_dir_path = annotations_dir
        self.example_dir = read_dir(example_dir)
        self.annotations_dir = [x.replace(".npy","_5.json") for x in self.example_dir]
        self.seq_length = 256
        self.transform = transform
        self.target_transform = target_transform
        

    def __len__(self):
        return len(self.example_dir)

    def __getitem__(self, idx):
        ds = load_npy(self.example_dir_path  + self.example_dir[idx])
        ds = torch.as_tensor(ds)
        # truth = read_meta(self.example_dir_path  + self.example_dir[idx].replace(".npy",".meta"))
        # ann = load_json(self.annotations_dir_path + self.annotations_dir[idx])

        if self.transform:
            sv = self.transform(ds[0])
            x,y,z = sv.shape
            if z > 526:
                sv = sv[:,:,:526]



        
        # if self.target_transform:
        #     ann, _ , _ = self.target_transform(ann,truth)

        
        enc,dec = self.split_encoder_decoder(sv,self.seq_length)

        ann = copy.deepcopy(dec)

        return {'enc': enc, 'dec': dec, 'target' :ann[random.choice(np.arange(0,dec.shape[0]))] }
    
    def split_encoder_decoder(self, data, seq_length):
        encoder = data[:, :seq_length, :]
        decoder = data[:, seq_length:, :]

        return encoder, decoder
class Dataset(torch.utils.data.Dataset):
    def __init__(self,example_dir, annotations_dir, transform=None,target_transform=None):
        self.example_dir_path = example_dir
        self.annotations_dir_path = annotations_dir
        self.example_dir = read_dir(example_dir)
        self.annotations_dir = [x.replace(".npy","_5.json") for x in self.example_dir]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.example_dir)

    def __getitem__(self, idx):
        ds = load_npy(self.example_dir_path  + self.example_dir[idx])
        truth = read_meta(self.example_dir_path  + self.example_dir[idx].replace(".npy",".meta"))
        ann = load_json(self.annotations_dir_path + self.annotations_dir[idx])

        if self.transform:
            sv = self.transform(ds[0])

        if self.target_transform:
            ann,dates,selection = self.target_transform(ann,truth)

        return torch.as_tensor(sv), torch.as_tensor(ann) # dates,selection ]
    
def collate_fn(batch):
    batch = batch
    enc = torch.cat([x['enc'] for x in batch],dim=0)
    dec = torch.cat([x['dec'] for x in batch],dim=0)
    target = torch.cat([x['target'] for x in batch],dim=0)
    return {'enc': enc, 'dec': dec, 'target' :target}

def create_dataloader(example_dir,annotations_dir,bsz=4,transform=transform_sv,target_transform=transform_labels_json):
    ds = Dataset(example_dir,annotations_dir,transform=transform,target_transform=target_transform)
    dl = DataLoader(ds, batch_size=bsz, shuffle=True, num_workers=4,collate_fn=collate_fn)
    return dl

def create_synthetic_dataloader(example_dir,annotations_dir,bsz=8,transform=transform_sv,target_transform=transform_labels_json):

    ds = SyntheticDataset(example_dir,annotations_dir,transform=transform,target_transform=target_transform)
    dl = DataLoader(ds, batch_size=bsz, shuffle=True, num_workers=4,collate_fn=collate_fn)

    return dl

# if __name__ == "__main__":
    # dl = create_dataloader("ds/ds_labeled/","ds/labels_crimac_2021/",transform=transform_sv,target_transform=transform_labels_json)

    # for item in dl:
    #     print(item)
    #     break
