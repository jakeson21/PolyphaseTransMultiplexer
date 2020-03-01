import numpy as np
from scipy import signal
from typing import Callable


def make_window(num_taps):
    # root-hann window
    h = signal.hann(num_taps+1)
    h = h[:-1]
    h = np.sqrt(h)
    return h


class WolaProcessor:
    def __init__(self,
                 fun: Callable[[np.ndarray], np.ndarray],
                 hop_size: int = 1,
                 block_size: int = 10,
                 c_valued: bool = True):

        assert (hop_size < block_size)
        self.iscomplex = c_valued
        # assert (block_size % hop_size == 0)
        self.process_func = fun
        self.R = hop_size
        self.N = block_size
        # make window and remove window scaling effect
        self.window = make_window(self.N) / np.sqrt((block_size/hop_size/2.0))
        if self.iscomplex:
            self.input_buff = np.zeros((self.N,), dtype=np.complex64)
            self.wola = np.zeros((self.N,), dtype=np.complex64)
        else:
            self.input_buff = np.zeros((self.N,), dtype=np.float64)
            self.wola = np.zeros((self.N,), dtype=np.float64)

    def process(self, x):
        assert(x.size == self.N)

        # apply analysis window
        self.input_buff = x * self.window
        if not self.iscomplex:
            Xm = np.fft.rfft(self.input_buff)
            # spectral processing here
            Ym = self.process_func(Xm)
            # return to time-domain and apply synthesis window
            ym = np.fft.irfft(Ym)
        else:
            Xm = np.fft.fft(self.input_buff)
            # spectral processing here
            Ym = self.process_func(Xm)
            # return to time-domain and apply synthesis window
            ym = np.fft.ifft(Ym)

        ym *= self.window

        self.wola[:-self.R] = self.wola[self.R:]
        self.wola[-self.R:] = 0
        self.wola += ym
        return self.wola[:self.R]
