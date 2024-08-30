from equalizer import *
from transmux.utils import plot_response


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


Gdb = np.asarray([9., 9, 9, 9, 9, 9, 9, 9, -9, -9, -6, -6, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 0, 0, 0, 9, 9, 9, 9])
numsopt, densopt = acge3(Gdb)
plot_response(numsopt, densopt, fs=44100/2., scalex='log')
# for k in range(numsopt.shape[1]):
#     plot_response(numsopt[:, k], densopt[:, k], fs=44100, scalex='log')
