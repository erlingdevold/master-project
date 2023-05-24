
"""
Normalizer class,
    - normalize the data

    median filter application

    Adapter for data stream for inference in pipeline
    Process .nc xarrays either as patches or time series.

"""
import xarray as xr
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import lightning

from utils import load_dataset,simrad_cmap

class Normalizer:
    def __init__(self,input_dir):
        self.dir = input_dir
        
    def load_dataset(self,fname="ds/ds_unlabeled/2019847-D20190423-T165617.nc"):
        return load_dataset(fname)

def apply_median_filter(ds: xr.Dataset):
    """
    Apply median filter to ds
    """
    if ds is None:
        return
    sv = ds.sv.to_numpy()

    # fig,ax = plt.subplots(2,1)
    # ax[0].imshow(rotate_image(sv[0]),cmap=simrad_cmap,aspect='auto',interpolation='none',label='cropped')

    sv  = median_filter(sv,size=3)
    # ax[1].imshow(rotate_image(sv[0]),cmap=simrad_cmap,aspect='auto',interpolation='none',)
    # fig.savefig("median_test.png")
    ds.sv.data = sv
    return ds 

def median_filter(x,size=3):
    """
    Median filter
    """
    x_new = np.zeros_like(x)
    for i in nb.prange(x.shape[0]):
        for j in nb.prange(x.shape[1]):
            if i == 0 or j == 0 or i == x.shape[0] - 1 or j == x.shape[1] - 1:
                x_new[i,j] = x[i,j]
                continue
            x_new[i,j] = np.median(x[i-size:i+size,j-size:j+size])

    return x_new

def rotate_image(x):
    return np.flipud(np.rot90(x,1))

def crop_matrix_bottom(ds,plot=False,fn='sd',crop=0):
    """
    Crop matrix from bottom line
    Finds largest index, and masks out the rest
    """
    OFFSET = 2

    if ds is None:
        return
    
    bottom_data = ds['bottom'].data[0]
    range = ds['range'].data
    
    largest_bottom_data = np.max(bottom_data)
    index = np.argmax(range > largest_bottom_data )

    x = ds

    if crop:
        x = ds.isel(range=slice(0,index + OFFSET))
    

    for i, bottom in enumerate(bottom_data):
        if i == bottom_data.shape[0] - 1:
            continue
        try:
            biggest_index = np.argmax(x.range.data > bottom)
        except IndexError:
            continue

        x.sv[0,i,biggest_index:] = -90

    return x

def transform_sv(sv):
    """
    Transform sv data
    """

    sv = arrange_data(sv)

    # sv = np.log(sv)

    return sv

def divide_in_sequences(ds, sequence_len = 256 ):
    num_sequences = ds.shape[0]//sequence_len

    num_vectors_to_use = num_sequences * sequence_len

    print(ds.shape)
    ds = ds[0:num_vectors_to_use]

    print(ds.shape)

    return ds.reshape(num_sequences, sequence_len,ds.shape[1])


def arrange_data(ds : xr.Dataset,sequence_len = 256):
    """

    sequence based arrangement of data
    sv is a 3d matrix, where the first dimension is the 
        frequency dimension, and the second and third are the
        range and time dimensions respectively.


        
    TODO : implement for CNN inference as well
    """

    if ds is None:
        return
    
    
    sv = divide_in_sequences(ds,sequence_len*2)

    return sv
import os

if __name__ == "__main__":
    print("Normalizer.py")

    norm = Normalizer("ds/ds_unlabeled/")
    ds = norm.load_dataset()
    fig,ax = plt.subplots(2,1)

    ax[0].imshow(rotate_image(ds.sv.data[0]),aspect='auto',interpolation='none',label='uncropped')
    x = crop_matrix_bottom(ds,crop=0)
    ax[1].imshow(rotate_image(x.sv.data[0]),aspect='auto',interpolation='none',label='cropped')

    print('asd')
    plt.savefig("crops.png")



