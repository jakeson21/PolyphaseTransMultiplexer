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
        self.filter = self.makePolyphaseFilter(order=40)

        self.inputBuffer = None

    def __str__(self):
        d = dict()
        d['sampleRateHz'] = self.sampleRateHz
        d['channelBandwidthHz'] = self.channelBandwidthHz
        d['M'] = self.M
        d['filter'] = self.filter.shape
        return json.dumps(d)

    def makePolyphaseFilter(self, order):
        # desired = (1, 1, 0, 0)
        # f = 1/self.M
        # bands = (0, f, f, 1)
        # h = signal.firls(order+1, bands, desired)
        # h = h / np.sum(h)

        # h = signal.firwin(order+1, cutoff=self.channelBandwidthHz/6, nyq=self.sampleRateHz/2)
        h = signal.firwin(order+1, cutoff=1/self.M)
        h[abs(h) <= 1e-4] = 0.
        h = h/np.max(h)/self.M

        # plt.plot(20*np.log10(np.abs(np.fft.fft(h, n=h.size))))
        # plt.grid(True)
        # plt.show()

        # form Hk
        N: int = np.ceil(order/self.M)
        h.resize((int(N)*self.M,), refcheck=False)
        # Columns are filter paths
        Hk = np.reshape(h,(-1, self.M))
        return Hk

    def polyphase_downFIR(self, x):
        # reshape input in prep for polyphase filtering
        xi = np.append(np.zeros((2,), dtype=np.csingle), x)
        xi = np.reshape(xi, (-1, self.M))

        # reorder input columns to [0 M-1 M-2 ... 1]
        xi = np.fliplr(xi)
        xi = np.roll(xi, shift=1, axis=1)
        # Apply polyphase filter to input
        xf = np.zeros_like(xi)
        for n in range(0, self.M):
            xf[:,n] = signal.fftconvolve(xi[:,n], self.filter[:,n], mode='same')
        # X = np.fft.fft(xi, axis=0)
        # H = np.fft.fft(self.filter, n=xi.shape[0], axis=0)
        # Y = X*H
        # xf = np.fft.ifft(Y, axis=0)

        tail = np.array([xf[-1,0], 0, 0], ndmin=2)
        xf[1:,0] = xf[0:-1,0]
        xf[0,0] = 0
        xf = np.concatenate((xf, tail))

        return xf

    def process(self, data):
        # if data.size % self.M != 0:
        #     raise IndexError('Input length must ba a multiple of M [{}]'.format(self.M))
        y = self.polyphase_downFIR(data)
        output = np.fft.ifft(y, n=self.M, axis=1) * self.M
        return output


if __name__ == "__main__":
    channelHz = 1000/3
    rx = PolyphaseRxChannelizer(sampleRateHz=channelHz*3, channelBandwidthHz=channelHz)
    print(rx)

    # data = np.ones((4000,), dtype=np.csingle)
    t = np.arange(0, 4, step=1/rx.sampleRateHz)
    fc = np.linspace(-1*rx.sampleRateHz/2, 0, t.size)
    data = np.exp(2j*np.pi*t*fc)
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
