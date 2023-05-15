
import numpy as np
import xarray as xr
import numba as nb
import matplotlib.pyplot as plt
from time import time
# from dask.distributed import Client

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
    return calculate_haversine(lat,labels_lat,lon,labels_lon)

if __name__ == "__main__":
    lat1 = np.array([50.0]*1000)
    lon1 = np.array([50.0]*1000)

    lat2 = np.array([51.0]*1000000)
    lon2 = np.array([51.0]*1000000)
    
    lon1 = lon1[:, np.newaxis]
    lat1 = lat1[:, np.newaxis]
    start = time()
    calculate_haversine(lat1,lat2,lon1,lon2)
    print(time()-start)





