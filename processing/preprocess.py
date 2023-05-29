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

from collator import Collator

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
            

    def process_raw(self,fn : str,plot=True,to_disk=True,btm_file=False) -> None:
        print(f"Processing {fn}...")
        self.echosounder = self.initialize_echosounder(fn)

        self.echosounder.read_raw(fn)
        if btm_file:
            self.bottom_file = True
            self.echosounder.read_bot(fn.replace(".raw",".bot"))

        channels = self.echosounder.raw_data.keys()
        print(channels)
        self._data = self.echosounder.get_channel_data()


        datasets = []

        for element in self._data:
            data = self._data[element][0]
            if np.allclose(data.get_frequency(), 120000., atol=1e-2) :
                print("retreived 120kHz data")
                ds = self.process_sv(data)
                datasets.append(ds)
                # print("skipping, we only look at 128kHZ frequnecies")

           
        labels = xr.open_dataset('labelling/dca_labels_subset.nc')

        if plot:
            fig,ax = plt.subplots()

        for ds in datasets:
            if plot:
                self.plot_data(ds,fig=fig,ax=ax,name=fn)
        
        if to_disk:
            freq_idx = 128000
            self.to_ds(datasets[0],fn)
        return datasets

    def to_ds(self,ds,fn):
        ds.to_netcdf(f'ds/ds_unlabeled/{fn.split("/")[-1].split(".")[0]}.nc')

    def process_sv(self,data) -> list:
        # crimac processing
        _data = self._data
        try:
            data_calibration = data.get_calibration()
        except KeyError as error:
            data_calibration = None
            print(error)
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

        
        if sv.ping_time.shape[0] == data.motion_data.heave.shape[0]:
            # crimac
            heave = data.motion_data.heave
            pitch = data.motion_data.pitch
            roll = data.motion_data.roll
            heading = data.motion_data.heading
        else:
            times = data.motion_data.times
            ping_time = sv.ping_time
            print(ping_time)
            p_idx = np.searchsorted(times,ping_time, side="right") - 1
            heave = data.motion_data.heave[p_idx]
            pitch = data.motion_data.pitch[p_idx]
            roll = data.motion_data.roll[p_idx]
            heading = data.motion_data.heading[p_idx]

        if not self.bottom_file:
            threshold_sv = 10 ** (-31.0 / 10)
            heave_corrected_transducer_depth = heave + depth[0]
            pulse_duration = float(pulse_length)

            depth_ranges, indices = detect_bottom_single_channel(
                xr_sv[0], threshold_sv,  heave_corrected_transducer_depth, pulse_duration, minimum_range=10.
            )
            
            depth_ranges_back_step, indices_back_step = btm.back_step(xr_sv[0], indices, heave_corrected_transducer_depth, .001)
            

            bottom_depths = heave_corrected_transducer_depth + depth_ranges_back_step - .5 
            bottom_depths = np.nan_to_num(bottom_depths, nan=np.min(bottom_depths))
            bottom_depths = xr.DataArray(name='bottom_depth', data=bottom_depths, dims=['ping_time'],
                                    coords={'ping_time': xr_sv['ping_time']})
        
        else:
            bottom_depths = data.get_bottom()

        sv_ds = xr.Dataset(data_vars = dict(
            sv=(["freq", "ping_time", "range"], xr_sv.data),
            angle_alongship=(["freq","ping_time","range"], np.expand_dims(alongship.data, axis=0)),
            angle_athwartship=(["freq","ping_time","range"], np.expand_dims(athwartship.data, axis=0)),
            transducer_draft=(["freq","ping_time"], depth.data),
            bottom = (["freq", "ping_time"], np.expand_dims(bottom_depths.data, axis=0)),
            heave = (["freq","ping_time"], np.expand_dims(heave,axis=0)),
            pitch = (["freq","ping_time"], np.expand_dims(pitch,axis=0)),
            roll = (["freq","ping_time"], np.expand_dims(roll,axis=0)),
            heading = (["freq","ping_time"], np.expand_dims(heading,axis=0)),
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
    
    def plot_data(self,ds,fig=None, ax=None, lines=None,name=None):
        if ds is None:
            return
        if ax is None:
            ax = plt.gca()

        data = np.array(ds.sv.data[0])

        simrad_cmap = (LinearSegmentedColormap.from_list('simrad', simrad_color_table))
        simrad_cmap.set_bad(color='grey')

        x_ticks = ds.sv.ping_time.astype('float')
        y_ticks = ds.sv.range.astype('float')
        
        data = np.flipud(np.rot90(data, 1))
        # ds.sv[.plot(x='ping_time', y='range', col='freq', col_wrap=1, cmap=simrad_cmap, vmin=-0, vmax=-70, aspect=1, size=5, robust=True)
    
        im = ax.imshow(data, cmap=simrad_cmap, aspect='auto',vmin=-0,vmax=-70, interpolation='none',extent=[x_ticks[0],x_ticks[-1],y_ticks[-1],y_ticks[0]]) # wrong axis what

        if lines is not None:
            ax.hlines(lines[2:], x_ticks[0], x_ticks[-1], colors='red', linestyles='dashed', linewidth=.2)
    
        fig.colorbar(im, orientation='horizontal', pad=0.05, aspect=50)
        fig.savefig(f'imgs/crimac/echo_{name.split("/")[-1].split(".")[0]}.png')
        fig.clf()




if __name__ == "__main__":
    p = Processor()
    collator = Collator()
    collator.load_labels()

    nordtind_data : str = '/data/saas/Nordtind/ES80/ES80-120/es80-120--D20190925-T212551.raw'
    crimac_data : str ='processing/crimac_data/cruise data/2021/D20210811-T134411.raw'

    file = nordtind_data
    files = p.read_files('processing/crimac_data/sample_pipeline_output/cruise_data/2019/S2019847_PEROS_3317/ACOUSTIC/EK60/EK60_RAWDATA')
    '/home/erling/master/processing/crimac_data/sample_pipeline_output/cruise_data/2019/S2019847_PEROS_3317/ACOUSTIC/EK60/EK60_RAWDATA'
    # files = p.read_files('/data/saas/Nordtind/ES80/ES80-120')
    # files = p.read_files('/data/saas/Ek80FraSimrad')
    for file in files:
        example = p.process_raw(file,btm_file=True,to_disk=True)
        print(example)
        plt.clf()

    print("done")
