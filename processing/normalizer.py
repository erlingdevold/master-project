
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
    




def crop_matrix_bottom(ds):
    """
    Crop matrix from bottom line
    """

    if ds is None:
        return
    
    bottom_data = ds['bottom'].data[0]

    bottom_data = np.array(bottom_data)

    # bottom_data = bottom_data[~np.isnan(bottom_data)]

    # find index i in sv that is closest to bottom
    x = ds.sel(range=bottom_data,method="nearest")

    plt.imshow(x.sv.data[0])
    plt.savefig("test.png")

    # crop matrix








    return ds


if __name__ == "__main__":
    print("Normalizer.py")

    norm = Normalizer("ds/ds_unlabeled/")
    ds = norm.load_dataset()
    x = crop_matrix_bottom(ds)

