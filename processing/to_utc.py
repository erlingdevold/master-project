import datetime
import numpy as np
import xarray as xr

import json
from utils import load_json
"""
File for converting json labels to xarray datasets.
provides utilites for calculating delta time and converting to days
Used for calculating mse in model
"""

def to_utc(dt):
    """Convert a datetime object to UTC time."""
    return dt.astimezone(datetime.timezone.utc)

def from_string(string : str,fmt="%d.%m.%Y"):
    """Convert a string to a datetime object."""
    return datetime.datetime.strptime(string,fmt) 


def create_delta_time(truth :str, obj):
    """Create a delta time object."""
    if obj is []:
        return []
    truth = from_string(truth,fmt="D%Y%m%d")
    return [(to_utc(from_string(x)) - to_utc(truth)).days for x in obj] 

def dict_to_da(obj,key):
    return  xr.DataArray(np.array([obj[key]['weight']],dtype=np.float64,ndmin=2),
                        dims=['species','x'],
                        coords={'species':[key],
                                'x':[0],
                                }, 
                        )

def convert_json_labels_to_xr(path,truth=''):
    """Parse a json file."""

    obj = load_json(path)
    da_list = []
    date_da_list = []
    for key in obj:
        if truth != '':
            obj[key]['date']= create_delta_time(truth, obj[key]['date'])
        
        
        da = dict_to_da(obj,key)
        date_da = xr.DataArray([obj[key]['date']],
                        dims=['species',"x"],
                        coords={'species':[key],
                                'x':range(len(obj[key]['date'])),
                                }, 
                        )
        da_list.append(da)
        date_da_list.append(date_da)

    da = xr.concat(da_list,dim='species')
    ds_weight = da.to_dataset(name='weight')

    date_da = xr.concat(date_da_list,dim='species')
    ds_date = date_da.to_dataset(name='date')

    ds = xr.merge([ds_weight,ds_date])

    return ds

def load_zarr(path):
    """Load a zarr file."""
    return xr.open_zarr(path)

import os
def convert_dir(path,truth):
    """Convert a directory of json files to a xr dataset."""

    files = os.listdir(path )
    for file in files:
        if file.endswith('.json'):
            try:
                ds = convert_json_labels_to_xr(path+file,'')
            except Exception as e:
                print(f"Could not convert {file} : {e}")
                continue
            if not os.path.exists(path+"zarr/" + file.replace(".json",'.zarr')):
                ds.to_zarr(path+"zarr/"+file.replace(".json",'.zarr'))
            # return path +"zarr/"+ file.replace(".json",'.zarr')

    return None
def find_max_number_species_code(dir:str = 'ds/labels_crimac_2021/', T="_5"):
    """
    Find the maximum number of species code in the dataset, given threshold
    """
    max_species = 0
    existing_label = np.zeros(len(os.listdir(dir)))
    selection = set([])
    for i,file in enumerate(os.listdir(dir)):
        if file.endswith(T+".json"):
            ds = load_json(dir+file)
            max_species = max(max_species,len(ds.keys()))
            if len(ds.keys()) > 0:
                existing_label[i] = 1
                keys = set([*ds])
                selection = selection.union(keys)


    print(selection,len(selection))

    selection = list(selection)
    selection.sort()
    return len(selection),selection

if __name__ == "__main__":
    truth = '03.05.2020'

    # ds = convert_dir('ds/labels_crimac_2021/','') # no truth, want zarrs
    truth = np.datetime64('2019-04-26T15:28:10.470000000')
    t5 = find_max_number_species_code()
    t20 = find_max_number_species_code(T="_20")
    t10 = find_max_number_species_code(T="_10")
    print(t5,t10,t20)



