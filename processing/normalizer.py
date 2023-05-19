
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

from utils import load_dataset

class Normalizer:
    def __init__(self,input_dir):
        self.dir = input_dir
        pass
        
    def load_dataset(self,fname):
        return load_dataset(fname)
    




if __name__ == "__main__":
    print("Normalizer.py")

    norm = Normalizer("ds/ds_unlabeled/")

