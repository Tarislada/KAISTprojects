import fcwt
import numpy as np
import matplotlib.pyplot as plt

fs = 1000
n = fs*100 #100 seconds
ts = np.arange(n)

#Generate linear chirp
signal = np.sin(2*np.pi*((0.1+(2*ts)/n)*(ts/fs)))

f0 = 0.1 #lowest frequency
f1 = 5 #highest frequency
fn = 3000 #number of frequencies

morl = fcwt.Morlet(2.0)

linscales = fcwt.Scales(morl, fcwt.FCWT_LINFREQS, fs, f0, f1, fn)

nthreads = 8
use_optimization_plan = False
use_normalization = True
fcwt_obj = fcwt.FCWT(morl, nthreads, use_optimization_plan, use_normalization)
#initialize output array
output = np.zeros((fn,signal.size), dtype=np.complex64)
