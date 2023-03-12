import matplotlib.pyplot as plt
from echolab2.instruments import EK80
from echolab2.plotting.matplotlib import echogram
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap, Colormap
import numpy as np
import matplotlib.cm as cm

# Create an EK80 instance

ek80_120hz = EK80.EK80()


def format_datetime(x, pos=None):
            try:
                dt = x.astype('datetime64[ms]').astype('object')
                tick_label = dt.strftime("%H:%M:%S")
            except:
                tick_label = ''

            return tick_label


ek80_120hz.read_raw(['es80-120--D20191119-T013458.raw','es80-38--D20191119-T013701.raw'])

# Read the raw data file

# Get the first EK80 ping

print(ek80_120hz)
# raw_data_38 = ek80.get_channel_data()

# Create an echogram

def plot(data_obj,ylabel,fq_int=120000,threshold=[-100,-1]):
    data = np.flipud(np.rot90(data_obj.data,1))
    if ylabel is None:
        # We don't have a valid axis.  Just use sample number.
        yticks = np.arange(data.shape[0])
        y_label = 'sample'
    elif hasattr(data_obj ,ylabel):
        yticks = getattr(data_obj, ylabel)
        y_label = ylabel + ' (m)'
    else:
        # We don't have a valid axis.  Just use sample number.
        yticks = np.arange(data.shape[0])
        y_label = 'sample'

    _simrad_color_table = [(1, 1, 1),
                                (0.6235, 0.6235, 0.6235),
                                (0.3725, 0.3725, 0.3725),
                                (0, 0, 1),
                                (0, 0, 0.5),
                                (0, 0.7490, 0),
                                (0, 0.5, 0),
                                (1, 1, 0),
                                (1, 0.5, 0),
                                (1, 0, 0.7490),
                                (1, 0, 0),
                                (0.6509, 0.3255, 0.2353),
                                (0.4705, 0.2353, 0.1568)]
    _simrad_cmap = (LinearSegmentedColormap.from_list
                            ('Simrad', _simrad_color_table))
    _simrad_cmap.set_bad(color='grey')
    
    xticks = data_obj.ping_time.astype('float')




    img  = plt.imshow(data, aspect='auto', vmin=threshold[0],vmax=threshold[-1],interpolation='none',extent=[
                xticks[0], xticks[-1],yticks[-1],yticks[0] ], origin='upper',cmap=_simrad_cmap )

    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(
    format_datetime))
    plt.colorbar(img, ax=plt.gca())
    plt.title(f'Sv {fq_int} Hz')
    plt.grid(True,color='k')

def plot_data(fq, fq_int=38000,threshold=[-100,-1]):
    channel_data = fq.get_channel_data(fq_int)
    channel_data = channel_data[fq_int][0]

    print(channel_data)

    power = channel_data.get_power()

    cal_obj = channel_data.get_calibration()

    sv= channel_data.get_Sv(calibration=cal_obj)
    if hasattr(sv, 'range'):
        ylabel= 'range'
    elif hasattr(sv, 'depth'):
        ylabel= 'depth'
    else:
        ylabel= None
    plot(sv.data,ylabel)
    


plot_data(ek80_120hz, 120000,)

plt.show()



