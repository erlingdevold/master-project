import numpy as np
import xarray as xr
import numba as nb
import matplotlib.pyplot as plt

from utils import calculate_distance, calculate_haversine,calculate_haversine_unvectorized


DISTANCE_KM_THRESHOLD = 10. # KM threshold for labelling

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
        labels_lat_end, labels_lon_end = np.array(self.labels['Stopposisjon bredde'].data),np.array(self.labels['Stopposisjon lengde'].data)

        lat,lon =ds.lat.data[0,0], ds.lon.data[0,0]

        plt.scatter(ds.lat.data[0],ds.lon.data[0],c='r',label='Vessel')

        positions = []
        unique_positions_end = np.array([[],[]])

        lat_transect = np.array(ds.lat.data[0])
        lon_transect = np.array(ds.lon.data[0])

        distance_matrix = calculate_haversine_unvectorized(lat_transect[:,np.newaxis],labels_lat,lon_transect[:,np.newaxis],labels_lon)
        print(distance_matrix.shape)
        # find all indices where distance is less than threshold
        indices = np.argwhere(distance_matrix < DISTANCE_KM_THRESHOLD)
        print(indices.shape)

        for lat, lon in zip(ds.lat.data[0],ds.lon.data[0]):
            # labels = labels.dropna(dim='Startposisjon lengde',how='any')
            haversine_start = apply_haversine(lat,labels_lat,lon,labels_lon)
            haversine_end = apply_haversine(lat,labels_lat_end,lon,labels_lon_end)
            print(haversine_start.min(),haversine_end.min())
            print(haversine_start[(haversine_start < DISTANCE_KM_THRESHOLD)])
            
            start = self.labels.isel(dim_0=np.argwhere(haversine_start < DISTANCE_KM_THRESHOLD).flatten())   
            end = self.labels.isel(dim_0=np.argwhere(haversine_end < DISTANCE_KM_THRESHOLD).flatten())   
            # need to combine start and end dataset

            tuble = [start['Startposisjon bredde'].data,start['Startposisjon lengde'].data]
            tuble_end = [end['Stopposisjon bredde'].data,end['Stopposisjon lengde'].data]

            # unique_positions = np.unique(unique_positions_end,axis=1)

            try:
                unique_ids = start.groupby('Melding ID')
            except ValueError as error:
                print(error)
                continue

            unique_positions = np.append(unique_positions,tuble,axis=1)

            unique_positions = np.unique(unique_positions,axis=1)

        positions = np.array(unique_positions)
        # end_positions = np.array(unique_positions_end)

        if plot:
            plt.scatter(positions[0,:],positions[1,:],c='b',label='start positions')
            # plt.scatter(end_positions[0,:],end_positions[1,:],c='g', marker="o", label='end positions')

            plt.legend()
            plt.savefig(f"imgs/crimac/map_{fname.split('/')[-1].split('.')[0]}.png")
            plt.clf()

                # species_max_weight= []
                # for art in art_grouped.groups:
                #     max_weight_by_species = weight.isel(Species=art_grouped.groups[art]).max()


    def plot_lat_lon(self,ax,labels):
        """
        Plot lat lon
        """
        # fig, ax = plt.subplots()
        # ax.scatter(ds.lat.data[0],ds.lon.data[0],label='data')
        ax.scatter(labels['Startposisjon bredde'].data,labels['Startposisjon lengde'].data,label='labels')
        ax.legend()

        
