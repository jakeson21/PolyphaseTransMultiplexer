import numpy as np
from scipy import signal


def make_window(num_taps, oversample, beta):
    fn = 1.235 / num_taps
    bands = (0, fn, fn, 1.0)
    desired = (1, 1, 0, 0)
    # num_taps, beta = signal.kaiserord(order, width=transition_width / (0.5 * fs))

    # h = signal.firwin(num_taps+1, cutoff=fn, window=('kaiser', beta), scale=False, fs=1)
    # h = h[:-1]
    # h = h / np.sum(h)

    h = signal.firls(oversample*num_taps+1, bands, desired, fs=2)
    hk = signal.kaiser(oversample*num_taps+1, beta=beta)
    h = h * hk
    h = h[:-1]
    h = h / np.sum(h)

    # h = signal.hann(num_taps+1)
    # h = h[:-1]
    # h = h / np.sum(h)

    return h


class WolaChannelizer:
    def __init__(self, sample_rate_hz: float, channels: int, transition=0.499):
        if channels > 1 and channels % 2 != 0:
            print('Must specify an even number of channels')
            return None
        self.sample_rate_Hz: float = sample_rate_hz
        self.min_bandwidth = self.sample_rate_Hz/channels
        self.R = int(channels/2)
        self.channel_bandwidth_Hz: float = self.sample_rate_Hz / channels
        self.t_last = 0
        self.overlap = 0.5
        self.oversample = 4
        self.window = make_window(2*self.R, self.oversample, beta=7)
        self.input_buff = np.zeros(int(self.oversample*2*self.R,), dtype=np.complex64)
        self.wola = np.zeros(int(self.oversample*2*self.R,), dtype=np.complex64)

        # NB Extraction parameters
        self.R2 = 5
        self.window2 = make_window(2*self.R2, self.oversample, beta=7)
        self.output_buff = np.zeros((self.oversample*2*self.R2,), dtype=np.complex64)
        self.wola_out = np.zeros((self.oversample*2*self.R2,), dtype=np.complex64)

    def process(self, x):
        assert(x.size == self.R)
        self.input_buff[:-self.R] = self.input_buff[self.R:]
        self.input_buff[-self.R:] = x

        self.wola = self.input_buff * self.window
        ws = self.wola.reshape(-1, 2*self.R)
        ws = np.sum(ws, axis=0)

        W = np.fft.fft(ws)
        # C = np.hstack((W[:self.R2], W[-self.R2:]))
        C = W
        C[0:4] = C[0:4]*0.010
        c = np.fft.ifft(C)

        s4 = np.asarray([c for n in range(self.oversample)]).reshape(-1, )
        self.output_buff[:-self.R2] = self.output_buff[self.R2:]
        self.output_buff[-self.R2:] = 0
        self.output_buff += s4 * self.window2
        x = self.output_buff[:self.R2]

        return x
