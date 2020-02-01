#!/usr/bin/env python3
import numpy as np
from scipy import signal
import json


class PolyphaseRxChannelizer:
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
        # outputBuffer stores the last M-1 filtered and reordered samples for use in the next iteration
        self.output_buffer = np.zeros((self.M - 1, 1), dtype=np.cdouble)

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
        # h[abs(h) <= 1e-4] = 0.
        # h = h/np.max(h)/self.M

        # form Hk
        N: int = int(np.ceil(np.ceil(h.size / self.M) * self.M))
        h.resize((N,), refcheck=False)
        # Columns are filter paths
        Hk = np.reshape(h, (-1, self.M)).transpose()
        return Hk

    def polyphase_down_fir(self, x):
        # reshape input in prep for polyphase filtering
        head_data_stream = np.asarray(x[::self.M])
        tail_data_streams = np.asarray([np.asarray(x[i::self.M]) for i in range(1, self.M)])
        xi = np.vstack((head_data_stream, tail_data_streams[::-1, :]))

        # Overlap and save here
        xf = np.asarray([signal.lfilter(self.filter[n, :], 1, np.hstack((self.input_buffer[n, :], xi[n, :]))) for n in
                         range(self.M)])
        # Save tail end for next time
        self.input_buffer = xi[:, -self.input_buffer.shape[1]:]
        # take second part
        xf = xf[:, self.input_buffer.shape[1]:]

        # App
        xd = np.asarray([np.append(xf[i], 0) if i == 0 else np.insert(xf[i], 0, 0) for i in range(self.M)])
        xd[1:, 0:1] = self.output_buffer
        self.output_buffer = xd[1:, -1:]

        return xd[:, :-1]

    def process(self, x):
        y = self.polyphase_down_fir(x)
        output = np.fft.ifft(y, n=self.M, axis=0) * self.M
        return output


if __name__ == "__main__":
    num_channels = 15
    chanBW = 50000
    Fs = chanBW * num_channels
    channelHz = Fs / num_channels
    rx = PolyphaseRxChannelizer(sample_rate_Hz=Fs, channel_bandwidth_Hz=channelHz, block_size=1000)
    print(rx)

