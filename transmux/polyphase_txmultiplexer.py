#!/usr/bin/env python3
import numpy as np
from scipy import signal
import json
from .utils import make_polyphase_filter


class PolyphaseTxMultiplexer:
    def __init__(self, sample_rate_hz: float, channels: int, transition=0.499):
        self.sample_rate_Hz: float = sample_rate_hz
        self.M = channels
        self.channel_bandwidth_Hz: float = self.sample_rate_Hz / self.M
        self.t_last = 0

        self.filter = make_polyphase_filter(self.M, transition_width=transition)
        # inputBuffer is used to save filterlength-1 WB samples for overlap-and-save based filtering
        self.input_buffer = np.zeros((self.M, self.filter.shape[1] - 1), dtype=np.cdouble)

    def __str__(self):
        d = dict()
        d['sample_rate_Hz'] = self.sample_rate_Hz
        d['channel_bandwidth_Hz'] = self.channel_bandwidth_Hz
        d['M'] = self.M
        d['filter'] = self.filter.tolist()
        return json.dumps(d)

    def polyphase_up_fir(self, x):
        # Filter by Overlap and save
        xf = np.asarray([signal.lfilter(self.filter[n, :], 1, np.hstack((self.input_buffer[n, :], x[n, :]))) for n in range(self.M)])
        # Save tail end for next time
        self.input_buffer = x[:, -self.input_buffer.shape[1]:]
        # take second part
        xf = xf[:, self.input_buffer.shape[1]:]
        # Up-sample: Commute output of filter stages at rate M by taking an output from every filter for each iteration
        return xf.transpose().reshape(-1, )

    def process(self, x):
        y = np.fft.ifft(x, n=self.M, axis=0) * np.sqrt(self.M)
        output = self.polyphase_up_fir(y)
        if self.M % 2 == 0:
            t = np.linspace(0, output.size, output.size, endpoint=True) + self.t_last
            self.t_last = t[-1] + t[1]-t[0]
            output *= np.exp(2j * np.pi / (2 * self.M) * t)
        return output


if __name__ == "__main__":
    num_channels = 3
    chanBW = 50000
    Fs = chanBW * num_channels
    channelHz = Fs / num_channels
    tx = PolyphaseTxMultiplexer(sample_rate_Hz=Fs, channel_bandwidth_Hz=channelHz)
    print(tx)
