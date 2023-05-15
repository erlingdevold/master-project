
import numpy as np
import xarray as xr
import numba as nb
import matplotlib.pyplot as plt
from time import time
# from dask.distributed import Client

@nb.njit(fastmath=True)
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
def calculate_haversine_unvectorized(lats_transect,lats_labels,lons_transect,lons_labels):
    i  =0
    lat_lon_tr = np.vstack((lats_transect,lons_transect))
    lat_lon_labels = np.vstack((lats_labels,lons_labels))
    print(lat_lon_tr.shape)

    array = np.zeros((lats_transect.shape[0],lons_labels.shape[0]))
    print(array.shape)

    for lat_lon in lat_lon_tr.T:
        j = 0
        lat_tr,lon_tr = lat_lon[0],lat_lon[1]

        for lat_lon_l in lat_lon_labels.T:
            lat_l,lon_l = lat_lon_l[0],lat_lon_l[1]

            km = calculate_haversine(lat_tr,lat_l,lon_tr,lon_l)
            # array[i][j] = km
            # j += 1


        i += 1
    return array


if __name__ == "__main__":
    lat1 = np.array([50.0]*1000)
    lon1 = np.array([50.0]*1000)

    lat2 = np.array([51.0]*1000000)
    lon2 = np.array([51.0]*1000000)
    
    start = time()
    calculate_haversine_unvectorized(lat1,lat2,lon1,lon2)
    print(time()-start)

    lon1 = lon1[:, np.newaxis]
    lat1 = lat1[:, np.newaxis]
    start = time()
    calculate_haversine(lat1,lat2,lon1,lon2)
    print(time()-start)





