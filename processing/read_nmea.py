import matplotlib.pyplot as plt
from echolab2.instruments import EK80
import numpy as np



ek80_38hx = EK80.EK80()

config = EK80.read_config('es80-38--D20191119-T013701.raw')

print(config.keys())


with open("test.txt", "w") as text_file:
    print(EK80.read_config('es80-38--D20191119-T013701.raw'), file=text_file)

# read motion data and nmea

