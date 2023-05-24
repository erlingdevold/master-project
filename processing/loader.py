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

class SyntheticEchoDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset=None,
                 seq_length=None,
                 normalize=False,
                 mean=None,
                 std=None):

        self.seq_lenght = seq_length
        self.normalize = normalize
        # self.data_utils = Data_utils()
        self.mean = mean
        self.std = std

        self.encoder, self.decoder = self.split_encoder_decoder(
            dataset, seq_length)

        # if normalize:
        #     # Normalize data
        #     self.encoder, self.decoder, self.targets = self.data_utils.normalize_dataset(
        #         self.encoder, self.decoder, self.mean, self.std)

    def split_encoder_decoder(self, data, seq_length):
        encoder = data[:, :seq_length, :]
        decoder = data[:, seq_length:, :]
        return encoder, decoder

    def __len__(self):
        return self.encoder.shape[0]

    def __getitem__(self, idx):
        sample = {
            'encoder': self.encoder[idx],
            'decoder': self.decoder[idx],
            # 'target':
            # self.targets[random.choice(np.arange(self.targets.shape[0]))]
        }

        return sample

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

        return torch.as_tensor(sv), [ann,] # dates,selection ]
    
def collate_fn(batch):
    return tuple(zip(*batch))

def create_dataloader(example_dir,annotations_dir,bsz=4,transform=transform_sv,target_transform=transform_labels_json):
    ds = Dataset(example_dir,annotations_dir,transform=transform,target_transform=target_transform)
    dl = DataLoader(ds, batch_size=bsz, shuffle=True, num_workers=0,collate_fn=collate_fn)
    return dl

if __name__ == "__main__":
    dl = create_dataloader("ds/ds_labeled/","ds/labels_crimac_2021/",transform=transform_sv,target_transform=transform_labels_json)
    for item in dl:
        print(item)
        break
