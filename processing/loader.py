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


import numpy as np

def apply_log( source):
    """
    Applies a log function to bring the dataset values down to a range 0-1
    :param source: The dataset to work on
    :return: The log version of the dataset
    """
    # Bring values down to range between 0 and 1
    source = torch.log(source)
    source[torch.isneginf(source)] = 0

    max_value = torch.max(source)
    source = source / torch.max(max_value)
    return source

def remove_examples_not_selection(dict,selection):
    for item in selection:
        if item not in dict:
            return False
        
    return True

def transform_labels_json(annotation : dict,truth:str,selection : list = None,percentages : bool = True,onehot : int = 1):
    if selection is None:
        raise Exception("Selection must be set during classification")

    sz = len(selection) 

    arr = np.zeros(sz)

    date_arr = []

    for key in selection:
        label = annotation.get(key,None)
        if label :
            arr[selection.index(key)] = 1 if onehot else label['weight']
            del annotation[key]
            date_arr.append(create_delta_time(truth, label['date']))
        else: # if no date, still insert empty list to ascertain right shape.
            date_arr.append([]) # no date for this label

    if not onehot and np.sum(arr) > 0:
        arr = np.clip(arr,1e-3,np.max(arr))
        arr = arr / np.sum(arr)
        arr = np.clip(arr,1e-3,1 - 1e-3)
    
    return arr, date_arr,selection

import random


def normalize_sv(sv):
    mean = torch.mean(sv)
    std = torch.std(sv)

    sv = (sv- mean)/std
            
    sv = sv / torch.max(sv)
    return sv
class Dataset(torch.utils.data.Dataset):
    def __init__(self,file_split,example_dir, annotations_dir,selection, transform=None,target_transform=None,classification_head = 2,task=0,threshold='_5', strict = True):
        self.head = classification_head
        self.selection= selection
        self.task = task

        self.file_split = file_split
        self.example_dir_path = example_dir
        self.annotations_dir_path = annotations_dir
        self.example_dir = read_dir(example_dir)
        for x in self.example_dir:
            if x.split('_')[1] not in self.file_split:
                self.example_dir.remove(x)
        if strict:
            for item in self.example_dir:
                if not remove_examples_not_selection(load_json(self.annotations_dir_path + item.split(".")[0].split('_')[1] + threshold + '.json'),selection):
                    self.example_dir.remove(item)

        self.annotations_dir = [x.split(".")[0].split('_')[1] + threshold + '.json'  for x in self.example_dir]
        print(f"size of ds {len(self.example_dir)} ")

        self.seq_length = 256
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.example_dir)

    def __getitem__(self, idx):

        ds = load_npy(self.example_dir_path  + self.example_dir[idx])
        ds = torch.as_tensor(ds)
        ann = load_json(self.annotations_dir_path + self.annotations_dir[idx])

        truth = self.example_dir[idx].split('_')[1].split('-')[1]

        if self.transform:

            sv = self.transform(ds[0])

            _,_,z = sv.shape
            if z > 526: # assert that we have the right dimension in example
                sv = sv[:,:,:526]

            sv = normalize_sv(sv)

        if self.target_transform:
            ann, dates , selection = self.target_transform(ann,truth,self.selection,onehot=self.task,percentages=True)

        
        enc,dec = self.split_encoder_decoder(sv,self.seq_length)

        return {'enc': enc, 'dec': dec, 'target' :torch.as_tensor(ann) , 'date' : dates, 'example' : self.example_dir[idx]}
    
    def split_encoder_decoder(self, data, seq_length):
        encoder = data[:, :seq_length, :]
        decoder = data[:, seq_length:, :]

        return encoder, decoder
 
class SyntheticDataset(Dataset):

    def __getitem__(self, idx):

        ds = load_npy(self.example_dir_path  + self.example_dir[idx])
        ds = torch.as_tensor(ds)

        if self.transform:
            sv = self.transform(ds[0])

            x,y,z = sv.shape
            if z > 526:
                sv = sv[:,:,:526]
            sv = normalize_sv(sv)
 
        
        enc,dec = self.split_encoder_decoder(sv,self.seq_length)

        ann = copy.deepcopy(dec)
        return {'enc': enc, 'dec': dec, 'target' :ann[random.choice(np.arange(0,dec.shape[0]))] }

def collate_fn_classifier(batch):
    batch = batch
    enc = torch.cat([x['enc'] for x in batch],dim=0)
    dec = torch.cat([x['dec'] for x in batch],dim=0)
    target = torch.stack([x['target'] for x in batch],dim=0)
    date = [x['date'] for x in batch]
    example = [x['example'] for x in batch]

    return {'enc': enc, 'dec': dec, 'target' :target, 'date' : date, 'example' : example }

def collate_fn2(batch):
    batch = batch
    enc = torch.cat([x['enc'] for x in batch],dim=0)
    dec = torch.cat([x['dec'] for x in batch],dim=0)
    target = torch.cat([x['target'] for x in batch],dim=0)
    date = torch.cat([x['date'] for x in batch],dim=0)
    example = [x['example'] for x in batch]

    return {'enc': enc, 'dec': dec, 'target' :target, 'date' : date, 'example' : example}
def create_dataloader(file_split,example_dir,annotations_dir,selection=None,bsz=8,transform=transform_sv,target_transform=transform_labels_json,shuffle=True,onehot=False,threshold="_5"):
    ds = Dataset(file_split,example_dir,annotations_dir,selection=selection,transform=transform,target_transform=target_transform,task=onehot,threshold=threshold)
    dl = DataLoader(ds, batch_size=bsz, shuffle=shuffle, num_workers=4,collate_fn=collate_fn_classifier)
    return dl

def create_synthetic_dataloader(file_split,example_dir,annotations_dir,bsz=8,transform=transform_sv,target_transform=transform_labels_json,shuffle=True,threshold="_5"):

    ds = SyntheticDataset(file_split, example_dir,annotations_dir,None,transform=transform,target_transform=None)
    dl = DataLoader(ds, batch_size=bsz, shuffle=shuffle, num_workers=4,collate_fn=collate_fn2)

    return dl

# if __name__ == "__main__":
    # dl = create_dataloader("ds/ds_labeled/","ds/labels_crimac_2021/",transform=transform_sv,target_transform=transform_labels_json)

    # for item in dl:
    #     print(item)
    #     break
