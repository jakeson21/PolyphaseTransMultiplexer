import numpy as np
from scipy import signal
from equalizer import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from transmux.utils import gen_complex_chirp, gen_fdma, gen_complex_awgn
from transmux.utils import gen_real_chirp, gen_fdma, gen_complex_awgn, plot_response


# fs = 44100.
# fc = 125.*2.*2.
# fB = fc/2.
#
# G = 2.  # peak gain (linear)
# GB = 0.5  # bandwidth gain (linear)
# w0 = 2.*np.pi*fc/fs  # center frequency (rads/sample)
# B = 1.5*w0  # bandwidth (rads/sample)
#
# [num, den] = pareq(G, GB, w0, B)
# plot_response(num, den)


Gdb = np.asarray([6., 6, 6, 6, 3, 3, -3, -3, -5, -5, -6, -6, 0, 0, 0, 3, 3, 3, 6, 6, 6, 9, 9, 9, 0, 0, 0, 12, 12, 12, 12])
numsopt, densopt = acge3(Gdb)
