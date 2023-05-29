
import numpy as np
import xarray as xr
import numba as nb
from numba import prange
import matplotlib.pyplot as plt
from time import perf_counter
from matplotlib.colors import LinearSegmentedColormap, Colormap

from const import simrad_color_table
# from dask.distributed import Client

simrad_cmap = (LinearSegmentedColormap.from_list('simrad', simrad_color_table))
simrad_cmap.set_bad(color='grey')

def load_dataset(fname):
    return xr.open_dataset(fname,engine='netcdf4')

@nb.njit(fastmath=True,parallel=True)
def calculate_haversine(lat_transect,lat_labels,lon_transect,lon_labels):
    lon_transect,lat_transect,lon_labels,lat_labels = np.radians(lon_transect),np.radians(lat_transect),np.radians(lon_labels),np.radians(lat_labels)

    dlon = lon_labels - lon_transect
    dlat = lat_labels - lat_transect

    a = np.sin(dlat/2.0)**2 + np.cos(lat_transect) * np.cos(lat_labels) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))

    return 6367 * c


def apply_haversine(lat1 , lat2 , lon1, lon2 ):

    return xr.apply_ufunc(calculate_haversine,lat1,lat2,lon1,lon2, dask='parallelized', output_dtypes=[float])

def calculate_distance(lat,lon,labels_lat,labels_lon):

    return calculate_haversine_unvectorized(lat,labels_lat,lon,labels_lon)

@nb.njit(fastmath=True,parallel=True)
def calculate_haversine_unvectorized(lats_transect,lats_labels,lons_transect,lons_labels,threshold=10.):
    i  =0

    array = np.zeros((lats_transect.shape[0],lons_labels.shape[0]))
    print(array.shape)
    print("starting")

    for i in prange(array.shape[0]):
        lat_i,lon_i = lats_transect[i],lons_transect[i]
        for j in prange(array.shape[1]):
            lat_j,lon_j = lats_labels[j],lons_labels[j]
            km = calculate_haversine(lat_i,lat_j,lon_i,lon_j)

            array[i][j] = km

    print("Threshold: ",threshold)
    indexes = np.argwhere(array < threshold) 

    return array, indexes

def convert_to_unique_indexes(indices,axis=0):
    """
    Convert indices to unique indexes
    """
    return np.unique(indices[:,axis])

def from_nc_to_zarr(dir):
    """
    Parse a dir to zarr arrays
    """

    import os
    files = os.listdir(dir)
    for file in files:
        if file.endswith('.nc'):
            try:
                ds = load_dataset(dir+file)
                ds.to_zarr(dir+'zarr/' +file.split(".")[0]+".zarr")
            except Exception as e:
                print(f"Could not convert {file} : {e}")
                continue

def segment_image(sv,segment_size=512):
    """
    Create patches of size segment_size from sv
    """

    return np.array_split(sv,sv.shape[1]// segment_size,axis=1)


def load_npy(path):
    """Load a npy sv"""
    return np.load(path)

def segment_dir(dir):
    """
    Parse a dir to zarr arrays
    """

    import os

    files = os.listdir(dir)

    for file in files:
        if file.endswith('.npy'):
            try:
                sv2 = segment_image(load_npy(dir+file)  )
                for i in range(len(sv2)-1):
                    discard_last = sv2[i].shape[1] < 512
                    if discard_last:
                        continue
                    segment = sv2[i][:,:512,:]
                    print(segment.shape)
                    np.save(dir+'segmented/' +f"{i}_"+file.split(".")[0]+".npy",segment)


            except Exception as e:
                print(f"Could not convert {file} : {e}")
                continue


def create_data_error(x):

    gaussian_function = lambda x, mu, sig: np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    return gaussian_function(x,0,10)

import os




if __name__ == "__main__":
    segment_dir('ds/ds_labeled/')
    # lat1 = np.array([51.0,71,51,51])
    # lon1 = np.array([51.0,71.0,51,51])

    # lat2 = np.array([51.0]*10)
    # lon2 = np.array([51.0]*10)

    # y_hat = np.ones((10,))
    # y = np.random.randint(0,10000,(10,))
    # #squeeze y and y_hat between 0 and 1
    # y_hat = y_hat / np.max(y_hat)
    # y = y / np.max(y)



    # dates_y = [np.random.randint(-1000,1000,np.random.randint(0,10))  for _ in range(10) ]
    # mse = (y_hat - y)**2

    # print(mse)
    # print(dates_y[1])
    # print(create_data_error(dates_y[1]))
    # print(create_data_error(dates_y[1] * -1) )

    # plt.plot(dates_y[1],create_data_error(dates_y[1]))

    # plt.savefig("mse.png")







