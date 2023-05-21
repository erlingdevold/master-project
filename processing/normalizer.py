
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

from utils import load_dataset,simrad_cmap

class Normalizer:
    def __init__(self,input_dir):
        self.dir = input_dir
        pass
        
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

@nb.njit
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

def crop_matrix_bottom(ds,plot=False,fn=''):
    """
    Crop matrix from bottom line
    Finds largest index, and masks out the rest
    """

    if ds is None:
        return
    
    bottom_data = ds['bottom'].data[0]

    largest_bottom_data = np.max(bottom_data)
    # crop on max value

    index = np.argmax(ds.range.data > largest_bottom_data )
    print(index)

    x = ds.isel(range=slice(None,index+2))

    for i, bottom in enumerate(bottom_data):
        if i == bottom_data.shape[0] - 1:
            continue
        try:
            biggest_index = np.argmax(x.range.data > bottom)
        except IndexError:
            continue
        x.sv[0,i,biggest_index:] = -70
    
    if plot:
        _,ax = plt.subplots(2,1)

        ax[0].imshow(rotate_image(x.sv.data[0]),cmap=simrad_cmap,aspect='auto',interpolation='none')
        ax[1].imshow(rotate_image(ds.sv.data[0]),cmap=simrad_cmap,aspect='auto',interpolation='none')

        plt.savefig(f"{fn}.png")
        plt.clf()

    return x

def arrange_data(ds : xr.Dataset):
    """

    sequence based arrangement of data
    sv is a 3d matrix, where the first dimension is the 
        frequency dimension, and the second and third are the
        range and time dimensions respectively.


        
    TODO : implement for CNN inference as well
    """

    if ds is None:
        return
    
    sv = ds.sv.to_numpy()







if __name__ == "__main__":
    print("Normalizer.py")

    norm = Normalizer("ds/ds_unlabeled/")
    ds = norm.load_dataset()
    x = crop_matrix_bottom(ds)

    x = apply_median_filter(x) # needs fixed up

    arrange_data(x)



