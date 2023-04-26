import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Colormap
from matplotlib import colors, ticker
from const import MAIN_FREQ,simrad_color_table
import xarray as xr
import pandas as pd
import random
import math

from echolab2.instruments import EK80, EK60
import os

from CRIMAC.bottomdetection import simple_bottom_detector as btm
from CRIMAC.bottomdetection.bottom_utils import detect_bottom_single_channel

import scipy.signal as signal

DISTANCE_KM_THRESHOLD = 1. # KM threshold for labelling

def format_datetime(x, pos=None):
            try:
                dt = x.astype('datetime64[ms]').astype('object')
                tick_label = dt.strftime("%H:%M:%S")
            except:
                tick_label = ''

            return tick_label


class Processor:
    def __init__(self) -> None:
        self.echosounder = None
        self._data = None
        self._nmea_data = None
        self._config = None

    def read_files(self,dir : str = ""):
        """
        reads a directories .raw files
        """
        files = os.listdir(dir)

        return [dir + "/" +file for file in files if file.endswith('.raw')]

    def initialize_echosounder(self,fn : str) -> None:
        with open(fn, 'rb') as f:
            header = f.read(8)
            magic = header[-4:]
            if magic.startswith(b'XML'):
                return  EK80.EK80()
            else:
                return EK60.EK60()
            

    def process_raw(self,fn : str) -> None:
        print(f"Processing {fn}...")
        self.echosounder = self.initialize_echosounder(fn)

        self.echosounder.read_raw(fn)

        channels = self.echosounder.raw_data.keys()
        print(channels)
        self._data = self.echosounder.get_channel_data()

        # try:
        #     self._data = self.echosounder.get_channel_data(frequencies=[MAIN_FREQ])[MAIN_FREQ][0]
        # except KeyError as error:
        #     print(f"{fn} does not contain data for {MAIN_FREQ} Hz: {error}")

        #     self._data = self.echosounder.get_channel_data()

        #     keys = [*self._data]
        #     self._data = self._data[keys[1]][0]
        #     for key in keys:
        #         print()

        #     print(self._data)
        FREQ_38KHZ = 38000
        datasets = []
        for element in self._data:
            data = self._data[element][0]
            if np.allclose(data.get_frequency(), FREQ_38KHZ, atol=1e-2) :
                print("skipping, we only look at higher frequnecies")
                continue

            ds = self.process_sv(data)
            datasets.append(ds)
            # except Exception as e:
            #     print(e)
            #     continue

        
        labels = xr.open_dataset('labelling/dca_labels_subset.nc')

        for ds in datasets:
            # self.plot_data(ds,lines=ds.bottom.data[0],name=f"imgs/{random.random()}{ds.freq.data[0]}")
            self.plot_data(ds,name=fn)
            self.collate_data(ds,labels,fname=fn.split("/")[-1].split(".")[0])


        # data_calibration = self._data.get_calibration()

        # sv = None
        # try:
        #     sv = self._data.get_Sv(calibration=data_calibration)
        # except Exception as error:
        #     print(error)
        
        # self._nmea_data = self.get_nmea_data(sv)
        # # self.plot_data(sv)
        # s_v = self._data.get_sv(calibration=data_calibration)
        # print(s_v)


    def collate_data(self,ds,labels,fname=""):
        """

        Collate lat,lon from labels to ds
        """
        if ds is None:
            return
        labels = labels.dropna(dim='dim_0',how='any')
        labels_lat, labels_lon = labels['Startposisjon bredde'].data,labels['Startposisjon lengde'].data

        # print(ds,labels)
        lat,lon =ds.lat.data[0,0], ds.lon.data[0,0]
        fig,ax = plt.subplots()

        ax.scatter(ds.lat.data[0],ds.lon.data[0],c='r',label='Ping')
        positions = []
        unique_positions = np.array([[],[]])

        for lat, lon in zip(ds.lat.data[0],ds.lon.data[0]):
            # labels = labels.dropna(dim='Startposisjon lengde',how='any')
            haversine = self.calculate_haversine(lat,lon,labels_lat,labels_lon)
            
            x = labels.isel(dim_0=np.argwhere(haversine < DISTANCE_KM_THRESHOLD).flatten())   
            # self.plot_lat_lon(ax,x)
            tuble = [x['Startposisjon bredde'].data,x['Startposisjon lengde'].data]

            unique_positions = np.append(unique_positions,tuble,axis=1)

            unique_positions = np.unique(unique_positions,axis=1)

            try:
                unique_ids = x.groupby('Melding ID')
            except ValueError as error:
                print(error)
                continue


            # for id in unique_ids.groups:
            #     unique_labels = labels.isel(dim_0=unique_ids.groups[id])
            
            #     data_vars = ['Rundvekt', 'Art FAO', 'Startposisjon bredde', 'Startposisjon lengde', 'Sluttposisjon bredde', 'Stopposisjon lengde']
            #         # new dataset with melding id as a dim
            #     weight = xr.DataArray(unique_labels['Rundvekt'].data, coords={"Species":unique_labels['Art FAO'].data} ,dims=['Species'], name=id)
            #     time = xr.DataArray(unique_labels['Meldingstidspunkt'].data , name=id)

            #     # print(time.data[0], ds.ping_time.data[0])

            #     art_grouped = weight.groupby("Species").max()
                # print(art_grouped)
        
        

        positions = np.array(unique_positions)
        # lats = xr.concat(positions[0,:],dim='dim_0')
        # lons = xr.concat(positions[1,:],dim='dim_0')
        ax.scatter(positions[0,:],positions[1,:],c='b',label='Labels')
        plt.legend()
        plt.savefig(f"imgs/map{fname}.png")

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
        

    
    def calculate_haversine(self,lat1,lon1,lat2,lon2):

        lon1,lat1,lon2,lat2 = map(np.radians, [lon1,lat1,lon2,lat2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

        c = 2 * np.arcsin(np.sqrt(a))

        return 6367 * c



    def process_sv(self,data) -> list:
        # crimac processing
        # try:
        _data = self._data
        try:
            data_calibration = data.get_calibration()
        except KeyError as error:
            data_calibration = None
            print(error)
        # except KeyError as error:
        #     data_calibration = None
        #     print(f"Error: {error}")

        try:
            sv = data.get_Sv(calibration = data_calibration)
        except Exception as error:
            sv = None
            print(error)
        if sv is None:
            return None

        s_v = data.get_sv()

        sv.set_navigation(data.nmea_data)
        print(sv.latitude,sv.longitude)

        data_3d = np.expand_dims(sv.data, axis=0)
        xr_sv = xr.DataArray(name="sv", data=data_3d, 
                            dims=["freq", "ping_time", "range"],
                            coords={"freq": [sv.frequency],
                                    "ping_time": sv.ping_time, 
                                    "range": sv.range,
                                    })
        # # add navigation to xarray



        depth = xr.DataArray(name="transducer_draft", data=np.expand_dims(sv.transducer_offset, axis=0), 
                            dims=['freq', 'ping_time'],
                            coords={ 'freq': [sv.frequency],
                                    'ping_time': sv.ping_time,
                                   })
        
        pulse_length = None
        angle_alongship = None
        angle_athwartship = None

        if hasattr(self._data, 'pulse_length'):
            pulse_length = np.unique(data.pulse_length)[0]
        elif hasattr(self._data, 'pulse_duration'):
            pulse_length = np.unique(data.pulse_duration)[0]
        else:
            pulse_length = 0

        try:
            alongship, athwartship = self.get_angles(data)
        except AttributeError:
            # Continuous wave sample
            alongship = np.zeros(sv.shape) * np.nan
            athwartship = np.zeros(sv.shape) * np.nan

        
        position_ds = xr.Dataset(data_vars = dict(
            lat=('ping_time' , sv.latitude), 
            lon=("ping_time",sv.longitude), 
            distance_nmi=('ping_time' , sv.trip_distance_nmi),
        ),
        coords = dict(ping_time=('ping_time', sv.ping_time))
        )



        
        if sv.ping_time.shape[0] == data.motion_data.heave.shape[0]:
            # crimac
            heave = data.motion_data.heave
        else:
            p_idx = np.searchsorted(data.motion_data.times, sv.ping_time.data, side="right") - 1
            heave = data.motion_data.heave[p_idx]

        heave = xr.DataArray(name="heave", data=np.expand_dims(heave, axis=0),
                            dims=['freq','ping_time'],
                            coords={'ping_time': sv.ping_time,
                                    'freq': [sv.frequency],
                                    }
                            )


        threshold_sv = 10 ** (-31.0 / 10)

        heave_corrected_transducer_depth = heave[0] + depth[0]
        pulse_duration = float(pulse_length)

        depth_ranges, indices = detect_bottom_single_channel(
            xr_sv[0], threshold_sv,  heave_corrected_transducer_depth, pulse_duration, minimum_range=10.
        )
        
        depth_ranges_back_step, indices_back_step = btm.back_step(xr_sv[0], indices, heave_corrected_transducer_depth, .001)
        

        bottom_depths = heave_corrected_transducer_depth + depth_ranges_back_step - .5 
        bottom_depths = np.nan_to_num(bottom_depths, nan=np.min(bottom_depths))
        bottom_depths = xr.DataArray(name='bottom_depth', data=bottom_depths, dims=['ping_time'],
                                 coords={'ping_time': xr_sv['ping_time']})
        

        sv_ds = xr.Dataset(data_vars = dict(
            sv=(["freq", "ping_time", "range"], xr_sv.data),
            angle_alongship=(["freq","ping_time","range"], np.expand_dims(alongship.data, axis=0)),
            angle_athwartship=(["freq","ping_time","range"], np.expand_dims(athwartship.data, axis=0)),
            transducer_draft=(["freq","ping_time"], depth.data),
            bottom = (["freq", "ping_time"], np.expand_dims(bottom_depths.data, axis=0)),
            heave = (["freq","ping_time"], np.expand_dims(data.motion_data.heave,axis=0)),
            pitch = (["freq","ping_time"], np.expand_dims(data.motion_data.pitch,axis=0)),
            roll = (["freq","ping_time"], np.expand_dims(data.motion_data.roll,axis=0)),
            heading = (["freq","ping_time"], np.expand_dims(data.motion_data.heading,axis=0)),
            pulse_length = (["freq"], [pulse_length]),
            lat = (["freq","ping_time"], np.expand_dims(sv.latitude,axis=0)),
            lon = (["freq","ping_time"], np.expand_dims(sv.longitude,axis=0)),
            distance = (["freq","ping_time"], np.expand_dims(sv.trip_distance_nmi,axis=0))
            ),
            coords = dict(
                freq = (["freq"], [sv.frequency]),
                ping_time = (["ping_time"], sv.ping_time),
                range = (["range"], sv.range),
            )
        )


        return sv_ds


    def get_angles(self,data):
        """
        Get the physical angles of the beams in sample.
        returns [alongship, athwartship]
        """
        try:
            return data.get_physical_angles(calibration=data.get_calibration())
        except AttributeError as e:
            print(e)
            return [np.zeros(data.shape) * np.nan, np.zeros(data.shape) * np.nan]

    def get_nmea_data(self,sv,data):
        return [data.nmea_data.interpolate(sv, attr) for attr in ['position','speed','distance']]
    
    def plot_data(self,ds, lines=None,name=None):
        if ds is None:
            return

        data = np.array(ds.sv.data[0])

        simrad_cmap = (LinearSegmentedColormap.from_list('simrad', simrad_color_table))
        simrad_cmap.set_bad(color='grey')

        x_ticks = ds.sv.ping_time.astype('float')
        y_ticks = ds.sv.range.astype('float')
        
        data = np.flipud(np.rot90(data, 1))
        # ds.sv[.plot(x='ping_time', y='range', col='freq', col_wrap=1, cmap=simrad_cmap, vmin=-0, vmax=-70, aspect=1, size=5, robust=True)
    
        im = plt.imshow(data, cmap=simrad_cmap, aspect='auto',vmin=-0,vmax=-70, interpolation='none',extent=[x_ticks[0],x_ticks[-1],y_ticks[-1],y_ticks[0]]) # wrong axis what

        if lines is not None:
            plt.hlines(lines[2:], x_ticks[0], x_ticks[-1], colors='red', linestyles='dashed', linewidth=.2)
    
        plt.colorbar(im, orientation='horizontal', pad=0.05, aspect=50)
        plt.savefig(f'imgs/asd{name.split("/")[-1].split(".")[0]}.png')
        plt.clf()



if __name__ == "__main__":
    p = Processor()

    nordtind_data : str = '/data/saas/Nordtind/ES80/ES80-120/es80-120--D20190925-T212551.raw'
    crimac_data : str ='processing/crimac_data/cruise data/2021/D20210811-T134411.raw'

    file = nordtind_data
    files = p.read_files('processing/crimac_data/cruise data/2021')
    # files = p.read_files('/data/saas/Nordtind/ES80/ES80-120')
    # files = p.read_files('/data/saas/Ek80FraSimrad')
    for file in files:
        p.process_raw(file)
        plt.clf()

    print("done")
