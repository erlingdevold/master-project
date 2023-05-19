import numpy as np
import xarray as xr
import numba as nb
import matplotlib.pyplot as plt
from time import perf_counter
import json

from utils import calculate_haversine_unvectorized,convert_to_unique_indexes, load_dataset


DISTANCE_KM_THRESHOLD = 10. # KM threshold for labelling
use_example = False

class Collator:

    def __init__(self):
        self.labels = None

    
    
    def load_labels(self,fname='labelling/dca_labels_subset.nc'):
        """
        Load labels from database
        """
        self.labels = xr.open_dataset(fname)

        self.labels = self.labels.dropna(dim='dim_0',how='any')



    def collate(self,ds,fname="",plot=False):
        """

        Collate lat,lon from labels to ds
        """
        if ds is None:
            return

        labels_lat, labels_lon = np.array(self.labels['Startposisjon bredde'].data),np.array(self.labels['Startposisjon lengde'].data)

        lat_transect = np.array(ds.lat.data[0])
        lon_transect = np.array(ds.lon.data[0])
        use_example = 0

        if not use_example:
            time = perf_counter()
            distance_matrix,indices = calculate_haversine_unvectorized(lat_transect,labels_lat,lon_transect,labels_lon,threshold=DISTANCE_KM_THRESHOLD)
            time_taken = perf_counter() - time
            print(f"Time taken: {time_taken}")

            indices = convert_to_unique_indexes(indices,axis=1)

        if use_example:
            selected_labels = xr.load_dataset("processing/example.nc")
        else:
            selected_labels = self.labels.isel(dim_0=indices)
        
        selected_labels = selected_labels.dropna(dim='dim_0',how='any')

        self.store_label_information(fname.split(".")[0],{"time": time_taken,"size" : len(selected_labels),"threshold":DISTANCE_KM_THRESHOLD})

        if plot:
            plt.scatter(ds.lat.data,ds.lon.data,label='Vessel')
            plt.scatter(selected_labels['Startposisjon bredde'].data,selected_labels['Startposisjon lengde'].data,label='Labels')
            plt.legend()
            plt.title(f"Labels for {fname.split('.')[0]}, Threshold: {DISTANCE_KM_THRESHOLD} km")
            plt.savefig(fname=f"imgs/labels/{fname.split('.')[0]}_{DISTANCE_KM_THRESHOLD}.svg")

        try:
            selected_labels_grouped = selected_labels.groupby('Melding ID')
        except Exception:
            return {}

        groups = selected_labels_grouped.groups

        dict = {}

        for group in groups:
            group_labels = selected_labels_grouped[group]
            for group_art_key, group_art_ds in list(group_labels.groupby("Art FAO")):
                if group_art_ds["Meldingsversjon"].data.max() != 1:
                    print("wow")
                if group_art_key not in dict:
                    dict[group_art_key] = {'weight':[],'date':[]}

                largest_version = group_art_ds.isel(dim_0=-1)

                dict[group_art_key]['weight'].append(largest_version["Rundvekt"].data)
                dict[group_art_key]['date'].append(str(largest_version["Startdato"].data))
                
        for art in dict:
            dict[art]['weight'] = np.sum(dict[art]['weight'])
            dict[art]["date"] = list(np.unique(dict[art]["date"]))

        return dict

    def write_to_file(self,ds,fname):
        fname = f"ds/labels/{fname}_labels.nc"
        if type(ds) == np.ndarray:
            ds = xr.DataArray(ds)

        ds.to_netcdf(fname)
    def store_labels(self,labels,fname):
        with open(f"ds/labels/{fname.split('.')[0]}_{DISTANCE_KM_THRESHOLD}.json", 'w') as fp:
            json.dump(labels, fp)

    def plot_lat_lon(self,ax,labels):
        """
        Plot lat lon
        """
        # fig, ax = plt.subplots()
        # ax.scatter(ds.lat.data[0],ds.lon.data[0],label='data')
        ax.scatter(labels['Startposisjon bredde'].data,labels['Startposisjon lengde'].data,label='labels')
        ax.legend()

    def label_example(self,fname):
        ds = load_dataset(fname)
        label_obj = self.collate(ds,fname=fname,plot=True)
        self.store_labels(label_obj,fname=fname)
    def store_label_information(self,fn,data):
        with open("ds/label_information_{fn}.json", 'w+') as fp:
            json.dump(data, fp)
            

        
import os

if __name__ == "__main__":
    c = Collator()
    c.load_labels()
    for file in os.listdir("ds/ds_unlabeled"):
        for threshold in [1,5,10,25,50,100]:
            DISTANCE_KM_THRESHOLD = threshold
            ds = load_dataset(f"ds/ds_unlabeled/{file}")

            label_obj = c.collate(ds,fname=file,plot=True)
            c.store_labels(label_obj,fname=file)
            plt.clf()



