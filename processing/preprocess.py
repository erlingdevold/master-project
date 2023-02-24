import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Colormap
from matplotlib import colors, ticker
from const import MAIN_FREQ,simrad_color_table
import xarray as xr
import pandas as pd

from echolab2.instruments import EK80


def format_datetime(x, pos=None):
            try:
                dt = x.astype('datetime64[ms]').astype('object')
                tick_label = dt.strftime("%H:%M:%S")
            except:
                tick_label = ''

            return tick_label

# pylint: disable=invalid-name, broad-exception-caught,unused_variable

class Processor:
    def __init__(self) -> None:
        self.ek80 = EK80.EK80()
        self._data = None
        self._nmea_data = None
        self._config = None

    def process_raw(self,fn : str) -> None:
        print(f"Processing {fn}...")
        self._config = EK80.read_config(fn)

        try:
            obj = self.ek80.read_raw(fn)
        except Exception as error:
            print(error)
        

        channels = self.ek80.raw_data.keys()
        print(channels)

        try:
            self._data = self.ek80.get_channel_data(frequencies=[MAIN_FREQ])[MAIN_FREQ][0]

        except KeyError as error:
            print(f"{fn} does not contain data for {MAIN_FREQ} Hz: {error}")

        data_calibration = self._data.get_calibration()

        sv = None
        try:
            sv = self._data.get_Sv(calibration=data_calibration)
        except Exception as error:
            print(error)
        
        print(sv)
        self._nmea_data = self.get_nmea_data(sv)
        power = self._data.get_power(calibration=data_calibration)
        self.plot_data(power)
        xr_sv, depth,pulse_len, ang1,ang2, pos = self.process_sv(sv)

        plt.show()

    def process_sv(self,sv ) -> list:
        # crimac processing
        sv.set_navigation(self._data.nmea_data)
        
        data_3d = np.expand_dims(sv.data, axis=0)
        xr_sv = xr.DataArray(name="sv", data=data_3d, 
                            dims=["freq", "ping_time", "range"],
                            coords={"freq": [sv.frequency],
                                    "ping_time": sv.ping_time, 
                                    "range": sv.range,
                                    })

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

        xr_sv.copy(data=np.expand_dims(alongship.data, axis=0))
        xr_sv.copy(data=np.expand_dims(athwartship.data, axis=0))
        print(xr_sv)
        print(self.get_nmea_data(sv))
        # self.plot_data(sv)

        return xr_sv, depth, pulse_length, angle_alongship, angle_athwartship, self.get_nmea_data(sv)

    def get_angles(self):
        """
        Get the physical angles of the beams in sample.
        returns [alongship, athwartship]
        """
        return self._data.get_physical_angles(calibration=self._data.get_calibration())

    def get_nmea_data(self,sv):
        return [self._data.nmea_data.interpolate(sv, attr) for attr in ['position','speed','distance']]
    
    def plot_data(self,sv):
        print(sv)
        data = sv.data

        simrad_cmap = (LinearSegmentedColormap.from_list('simrad', simrad_color_table))
        simrad_cmap.set_bad(color='grey')

        x_ticks = sv.ping_time.astype('float')
        y_ticks = sv.range.astype('float')
        
        data = np.flipud(np.rot90(data, 1))
        im = plt.imshow(data, cmap=simrad_cmap, aspect='auto', interpolation='none', extent=[x_ticks[0],x_ticks[-1],y_ticks[-1],y_ticks[0]]) # wrong axis what
    
        plt.colorbar(im, orientation='horizontal', pad=0.05, aspect=50)
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_datetime))



if __name__ == "__main__":
    p = Processor()

    p.process_raw('crimac_data/EK80examples/FM/CRIMAC_2020_EK80_FM_DemoFile_GOSars.raw')
    print("done")
