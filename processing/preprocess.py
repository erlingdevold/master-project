import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Colormap
from matplotlib import colors, ticker
from const import MAIN_FREQ,simrad_color_table
import xarray as xr
import pandas as pd

from echolab2.instruments import EK80, EK60
import os

from CRIMAC.bottomdetection import simple_bottom_detector as btm
from CRIMAC.bottomdetection.bottom_utils import detect_bottom_single_channel



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

        print(self.echosounder)
        obj = self.echosounder.read_raw(fn)
        channels = self.echosounder.raw_data.keys()

        try:
            self._data = self.echosounder.get_channel_data(frequencies=[MAIN_FREQ])[MAIN_FREQ][0]
        except KeyError as error:
            print(f"{fn} does not contain data for {MAIN_FREQ} Hz: {error}")
            return None

        data_calibration = self._data.get_calibration()

        sv = None
        try:
            sv = self._data.get_Sv(calibration=data_calibration)
        except Exception as error:
            print(error)
        
        self._nmea_data = self.get_nmea_data(sv)
        # self.plot_data(sv)
        s_v = self._data.get_sv(calibration=data_calibration)
        print(s_v)

        self.process_sv(sv)


    def process_sv(self,sv ) -> list:
        # crimac processing

        sv.set_navigation(self._data.nmea_data)
        print(sv.latitude,sv.longitude)

        # print(sv)
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
            pulse_length = np.unique(self._data.pulse_length)[0]
        elif hasattr(self._data, 'pulse_duration'):
            pulse_length = np.unique(self._data.pulse_duration)[0]
        else:
            pulse_length = 0

        alongship, athwartship = self.get_angles()
        
        position_ds = xr.Dataset(data_vars = dict(
            lat=('ping_time' , sv.latitude), 
            lon=("ping_time",sv.longitude), 
            distance_nmi=('ping_time' , sv.trip_distance_nmi),
        ),
        coords = dict(ping_time=('ping_time', sv.ping_time))
        )


        heave = self._data.motion_data.heave
        
        heave = xr.DataArray(name="heave", data=np.expand_dims(heave, axis=0),
                            dims=['freqs','ping_time'],
                            coords={'ping_time': sv.ping_time,
                                    'freqs': [sv.frequency],
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
            botttom = (["freq", "ping_time"], np.expand_dims(bottom_depths.data, axis=0)),
            heave = (["ping_time"], self._data.motion_data.heave),
            pitch = (["ping_time"], self._data.motion_data.pitch),
            roll = (["ping_time"], self._data.motion_data.roll),
            heading = (["ping_time"], self._data.motion_data.heading),
            pulse_length = (["freq"], [pulse_length]),
            lat = (["ping_time"], sv.latitude),
            lon = (["ping_time"], sv.longitude),
            distance = (["ping_time"], sv.trip_distance_nmi)
            ),
            coords = dict(
                freq = (["freq"], [sv.frequency]),
                ping_time = (["ping_time"], sv.ping_time),
                range = (["range"], sv.range),
            )
        )

        print(sv_ds)


    def get_angles(self):
        """
        Get the physical angles of the beams in sample.
        returns [alongship, athwartship]
        """
        return self._data.get_physical_angles(calibration=self._data.get_calibration())

    def get_nmea_data(self,sv):
        return [self._data.nmea_data.interpolate(sv, attr) for attr in ['position','speed','distance']]
    
    def plot_data(self,sv, lines=None):
        print(sv)
        data = np.array(sv.data)

        simrad_cmap = (LinearSegmentedColormap.from_list('simrad', simrad_color_table))
        simrad_cmap.set_bad(color='grey')

        x_ticks = sv.ping_time.astype('float')
        y_ticks = sv.range.astype('float')
        
        data = np.flipud(np.rot90(data, 1))
        im = plt.imshow(data, cmap=simrad_cmap, aspect='auto', interpolation='none', extent=[x_ticks[0],x_ticks[-1],y_ticks[-1],y_ticks[0]]) # wrong axis what

        plt.hlines(lines, x_ticks[0], x_ticks[-1], colors='r', linestyles='dashed')
    
        plt.colorbar(im, orientation='horizontal', pad=0.05, aspect=50)
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_datetime))
        plt.savefig('sv.png')



if __name__ == "__main__":
    p = Processor()

    nordtind_data : str = '/data/saas/Nordtind/ES80/ES80-120/es80-120--D20190925-T212551.raw'
    crimac_data : str ='crimac_data/cruise data/2021/D20210811-T134411.raw'

    data = crimac_data 
    files = p.read_files('crimac_data/cruise data/2021')
    for file in files:
        p.process_raw(file)

    print("done")
