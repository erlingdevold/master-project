
import numpy as np
import xarray as xr
import numba as nb
from numba import prange
import matplotlib.pyplot as plt
from time import perf_counter
# from dask.distributed import Client

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
    """
    Two vectors of shape 2xN, N is number of element in corresponding vector,
    N is different for both vectors

    calculate distance between each element in vector 1 and each element in vector 2
    can not broadcast
    """
    return calculate_haversine_unvectorized(lat,labels_lat,lon,labels_lon)

@nb.njit(fastmath=True,parallel=True)
def calculate_haversine_unvectorized(lats_transect,lats_labels,lons_transect,lons_labels,threshold=10.):
    i  =0
    lat_lon_tr = np.vstack((lats_transect,lons_transect))
    lat_lon_labels = np.vstack((lats_labels,lons_labels))
    print(lat_lon_tr.shape)

    array = np.zeros((lats_transect.shape[0],lons_labels.shape[0]))
    print(array.shape)
    print("starting")

    for i in prange(array.shape[0]):
        lat_i,lon_i = lats_transect[i],lons_transect[i]
        for j in prange(array.shape[1]):
            lat_j,lon_j = lats_labels[j],lons_labels[j]
            km = calculate_haversine(lat_i,lat_j,lon_i,lon_j)

            array[i][j] = km

    # time = perf_counter()
    print("Threshold: ",threshold)
    indexes = np.argwhere(array < threshold) 
    # print(f"Time taken: {perf_counter() - time}")

    return array, indexes

def convert_to_unique_indexes(indices,axis=0):
    """
    Convert indices to unique indexes
    """
    return np.unique(indices[:,axis])


if __name__ == "__main__":
    lat1 = np.array([51.0,71,51,51])
    lon1 = np.array([51.0,71.0,51,51])

    lat2 = np.array([51.0]*10)
    lon2 = np.array([51.0]*10)
    
    start = time()
    x,indices = calculate_haversine_unvectorized(lat1,lat2,lon1,lon2)
    # asd = np.unique(indices[:,0])
    asd = convert_to_unique_indexes(indices)
    #returns matrix of shape (lat1,lat2), indices of shape (lat1,lat2,2)
    # find all indices in lat2 where distance is less than threshold

    # indices = convert_to_unique_indexes(indices)

    print(time()-start)

    lon1 = lon1[:, np.newaxis]
    lat1 = lat1[:, np.newaxis]
    start = time()
    calculate_haversine(lat1,lat2,lon1,lon2)
    print(time()-start)





