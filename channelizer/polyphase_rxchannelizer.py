#!/usr/bin/env python3
from typing import Mapping, MutableMapping, Sequence, Iterable, List, Set, NewType, Dict
import numpy as np
from scipy import signal
import json
import pdb
import matplotlib.pyplot as plt


class PolyphaseRxChannelizer():
    def __init__(self, sampleRateHz: int, channelBandwidthHz: int):
        self.sampleRateHz: int = sampleRateHz
        self.channelBandwidthHz: int = channelBandwidthHz

        self.M = int(self.sampleRateHz / self.channelBandwidthHz)
        self.filter = self.makePolyphaseFilter(self.M)

        self.inputBuffer = None

    def __str__(self):
        d = dict()
        d['sampleRateHz'] = self.sampleRateHz
        d['channelBandwidthHz'] = self.channelBandwidthHz
        d['M'] = self.M
        d['filter'] = self.filter.shape
        return json.dumps(d)

    def makePolyphaseFilter(self, length: int):
        desired = (1, 1, 0, 0)
        f = 0.5
        bands = (0, f, f, 1)
        h = signal.firls(20*length+1, bands, desired)
        # h = h / np.sum(h)
        # form Hk
        h.resize((21*length,))
        # Columns are filter paths
        Hk = np.reshape(h,(-1,length))
        return Hk

    def polyphase_downFIR(self, x):
        # reshape input in prep for polyphase filtering
        xi = np.reshape(x, (-1, self.M))
        # reorder input columns to [0 M-1 M-2 ... 1]
        xf = signal.fftconvolve(xi, self.filter, mode='same')
        return xf

    def process(self, data):
        if data.size % self.M != 0:
            raise IndexError('Input length must ba a multiple of M [{}]'.format(self.M))
        y = self.polyphase_downFIR(data)
        output = np.fft.ifft(y, n=self.M, axis=1) * self.M
        return output


if __name__ == "__main__":
    channelHz = 1
    rx = PolyphaseRxChannelizer(sampleRateHz=channelHz*4, channelBandwidthHz=channelHz)
    print(rx)

    data = np.ones((rx.M*100,), dtype=np.csingle)
    fc = 2
    t = np.arange(0,data.size)/rx.sampleRateHz
    data *= np.exp(2j*np.pi*t*fc)
    # data = np.arange(0,rx.M*50, dtype=np.csingle)
    output = rx.process(data)
    print(output.shape)
    np.set_printoptions(precision=4, linewidth=180)
    print(output)

    for n in range(0, rx.M):
        plt.subplot(rx.M, 1, n+1)
        plt.plot(output[:,n])
        plt.grid(True)
    plt.show()
