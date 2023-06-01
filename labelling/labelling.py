from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from sqlalchemy.ext.declarative import declarative_base
import dca as dca
import xarray as xr

engine = create_engine('postgresql://postgres:example@localhost:5420/postgres', echo=True)

def create_select_query(items : list,limit = 0):
    query = "SELECT "
    for item in items:
        if item == "*":
            query = "SELECT *  "
        else:
            query += '"'+ item +'"'+ ", "
    query = query[:-2]
    query += " FROM dca" 
    query += f" LIMIT {limit}" if limit else ""
    return query

def get_columns():
    with engine.connect() as conn:
        res = conn.execute(text("select * from dca limit 1;"))  
    return res.keys()

def read_data():
    
    with engine.connect() as conn:
        res = conn.execute(text(create_select_query(["*"])))  
    return res

def read_data_to_xarray(data):
    
    columns = list(data.keys())

    xr_data = xr.DataArray(data.fetchall())

    xr_data = xr_data.assign_coords({"dim_1": columns})
    xr_data = xr_data.to_dataset(dim="dim_1")

    return xr_data

def write_to_file(data,fn="labelling/dca_labels.nc"):
    data.to_netcdf(fn)


def read_file(file):
    data = xr.open_dataset(file)
    return data

def create_subset(data,labels):
    return data[labels]

def merge_labels(labels):
    '''
    Merge 
    '''
    return labels

import os

if __name__ == "__main__":

    if not os.path.exists("labelling/dca_labels.nc"):
        data = read_data()
        labels = read_data_to_xarray(data)
        labels = merge_labels(labels)
        write_to_file(labels)

    if not os.path.exists("labelling/dca_labels_subset.nc"):
        labels =read_file("labelling/dca_labels.nc")

        all_labels =  list(labels.data_vars.keys())
        
        labels_to_select = ["Starttidspunkt","Startdato","Stopptidspunkt","Stoppdato","Startposisjon bredde","Havdybde start","Rundvekt","Havdybde stopp","Melding ID","Meldingsversjon","Meldingstidspunkt","Startposisjon lengde","Stopposisjon bredde","Stopposisjon lengde","Art FAO", "Hovedart FAO","Meldingsnummer","Art FAO (kode)","Hovedart FAO (kode)"]
        subset_labels = create_subset(labels,labels_to_select) 

        subset_labels = subset_labels.where(subset_labels["Startposisjon bredde"] != '') 
        subset_labels["Startposisjon bredde"] = subset_labels["Startposisjon bredde"].astype(str).str.replace(",",".").astype(float)
        subset_labels["Startposisjon lengde"] = subset_labels["Startposisjon lengde"].astype(str).str.replace(",",".").astype(float)
        subset_labels["Stopposisjon bredde"] = subset_labels["Stopposisjon bredde"].astype(str).str.replace(",",".").astype(float)
        subset_labels["Stopposisjon lengde"] = subset_labels["Stopposisjon lengde"].astype(str).str.replace(",",".").astype(float)

    #     # drop nan values
        subset_labels = subset_labels.dropna(dim="dim_0",how="any")

        duplicate_id_idx = subset_labels["Melding ID"]

        write_to_file(subset_labels,"labelling/dca_labels_subset.nc")

    labels = read_file("labelling/dca_labels_subset.nc")




