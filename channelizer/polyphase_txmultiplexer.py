#!/usr/bin/env python3
import numpy as np
from scipy import signal
import json
import pdb


class PolyphaseTxMultiplexer:
    def __init__(self, sample_rate_Hz: float, channel_bandwidth_Hz: float, block_size: int):
        self.sample_rate_Hz: float = sample_rate_Hz
        self.channel_bandwidth_Hz: float = channel_bandwidth_Hz
        self.blockSize = block_size

        # Ensure total bandwidth is evenly divisible by the channel bandwidth
        if (sample_rate_Hz / channel_bandwidth_Hz) != np.round(sample_rate_Hz / channel_bandwidth_Hz):
            raise ValueError(
                'sample_rate_Hz ({}) must be evenly divisible by channel_bandwidth_Hz ({})'.format(sample_rate_Hz,
                                                                                                   channel_bandwidth_Hz))

        self.M = int(self.sample_rate_Hz / self.channel_bandwidth_Hz)
        order = 10
        self.filter = self.make_polyphase_filter(order=order)
        # inputBuffer is used to save filterlength-1 WB samples for overlap-and-save based filtering
        self.input_buffer = np.zeros((self.M, self.filter.shape[1] - 1), dtype=np.cdouble)

    def __str__(self):
        d = dict()
        d['sample_rate_Hz'] = self.sample_rate_Hz
        d['channel_bandwidth_Hz'] = self.channel_bandwidth_Hz
        d['M'] = self.M
        d['filter'] = self.filter.tolist()
        return json.dumps(d)

    def make_polyphase_filter(self, order):
        fs = self.M
        # cutoff = 1.0/(self.M - 0.25*self.M)
        cutoff = 0.5
        transition_width = 0.1
        numtaps, beta = signal.kaiserord(150, width=transition_width / (0.5 * fs))
        print('fs={}, transition_width={}, cutoff={}, numtaps={}, beta={}'.format(fs, transition_width, cutoff, numtaps,
                                                                                  beta))

        h = signal.firwin(numtaps, cutoff=cutoff, window=('kaiser', beta), scale=False, nyq=0.5 * self.M)
        h[abs(h) <= 1e-15] = 0.
        h = h / np.max(abs(h))

        # form Hk
        N: int = int(np.ceil(np.ceil(h.size / self.M) * self.M))
        h.resize((N,), refcheck=False)
        # Columns are filter paths
        Hk = np.reshape(h, (-1, self.M)).transpose()
        return Hk

    def polyphase_up_fir(self, x):
        # Filter by Overlap and save
        xf = np.asarray([signal.lfilter(self.filter[n, :], 1, np.hstack((self.input_buffer[n, :], x[n, :]))) for n in
                         range(self.M)])
        # Save tail end for next time
        self.input_buffer = x[:, -self.input_buffer.shape[1]:]
        # take second part
        xf = xf[:, self.input_buffer.shape[1]:]
        # Up-sample: Commute output of filter stages at rate M by taking an output from every filter for each iteration
        return xf.transpose().reshape(-1, )

    def process(self, x):
        y = np.fft.ifft(x, n=self.M, axis=0) * self.M
        output = self.polyphase_up_fir(y)
        return output


if __name__ == "__main__":
    num_channels = 3
    chanBW = 50000
    Fs = chanBW * num_channels
    channelHz = Fs / num_channels
    tx = PolyphaseTxMultiplexer(sample_rate_Hz=Fs, channel_bandwidth_Hz=channelHz, block_size=1000)
    print(tx)
