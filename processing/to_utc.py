import datetime
import numpy as np
import xarray as xr

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

import json

def create_delta_time(truth :str, obj):
    """Create a delta time object."""
    truth = from_string(truth,fmt="%Y-%m-%dT%H")
    # obj = from_string(obj)
    return [(to_utc(from_string(x)) - to_utc(truth)).days for x in obj] 

def load_json(path):
    """Load a json file."""
    with open(path, "r") as f:
        return json.load(f)

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

def convert_dir(path,truth):
    """Convert a directory of json files to a xr dataset."""

    import os
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

if __name__ == "__main__":
    truth = '03.05.2020'

    # ds = convert_dir('ds/labels_crimac_2021/','') # no truth, want zarrs
    truth = np.datetime64('2019-04-26T15:28:10.470000000')

    np.datetime_as_string()
    x = load_zarr(ds)
    print(x)


