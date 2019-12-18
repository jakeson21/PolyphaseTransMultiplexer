# this module will be imported in the into your flowgraph
from scipy import signal
import numpy as np

def makePolyphaseFilter(M, order=10):
    fs = M
    # cutoff = 1.0/(self.M - 0.25*self.M)
    cutoff = 0.5
    transition_width = 0.1
    numtaps, beta = signal.kaiserord(150, width=transition_width/(0.5*fs))
    print('fs={}, transition_width={}, cutoff={}, numtaps={}, beta={}'.format(fs, transition_width, cutoff, numtaps, beta))

    h = signal.firwin(numtaps, cutoff=cutoff, window=('kaiser', beta), scale=False, nyq=0.5*fs)
    # h[abs(h) <= 1e-4] = 0.
    # h = h/np.max(h)/self.M

    # form Hk
    N = int(np.ceil(np.ceil(h.size/fs)*fs))
    h.resize((N,), refcheck=False)
    # Columns are filter paths
    # Hk = np.reshape(h,(-1, fs)).transpose()
    return h

