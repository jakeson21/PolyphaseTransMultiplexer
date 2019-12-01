#!/usr/bin/env python3
from typing import Mapping, MutableMapping, Sequence, Iterable, List, Set, NewType, Dict
import numpy as np
from scipy import signal
import json
import pdb
import matplotlib.pyplot as plt


class PolyphaseRxChannelizer():
    def __init__(self, sampleRateHz: float, channelBandwidthHz: float, blockSize: int):
        self.sampleRateHz: float = sampleRateHz
        self.channelBandwidthHz: float = channelBandwidthHz
        self.blockSize = blockSize

        if (sampleRateHz/channelBandwidthHz) != np.round(sampleRateHz/channelBandwidthHz):
            raise ValueError('sampleRateHz must be evenly divisible by channelBandwidthHz')

        self.M = int(self.sampleRateHz / self.channelBandwidthHz)
        Order = 40
        self.filter = self.makePolyphaseFilter(order=Order)

        self.inputBuffer = np.zeros((self.M, Order),dtype=np.csingle)
        self.outputBuffer = np.zeros((self.M-1, 1),dtype=np.csingle)

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
        h = signal.firwin(order*self.M, cutoff=1.0/(self.M))
        h[abs(h) <= 1e-4] = 0.
        h = h/np.max(h)/self.M

        # plt.plot(20*np.log10(np.abs(np.fft.fft(h, n=h.size))))
        # plt.grid(True)
        # plt.show()

        # form Hk
        N: int = int(np.ceil(np.ceil(h.size/self.M)*self.M))
        h.resize((N,), refcheck=False)
        # Columns are filter paths
        Hk = np.reshape(h,(-1, self.M)).transpose()
        return Hk

    def polyphase_downFIR(self, x):
        # reshape input in prep for polyphase filtering
        head_data_stream = np.asarray(x[::self.M])
        tail_data_streams = np.asarray([np.asarray(x[0+i::self.M]) for i in range(1,self.M)])
        xi = np.vstack((head_data_stream, tail_data_streams[::-1,:]))
        
        # Overlap and save here
        xf = np.asarray([signal.lfilter(self.filter[n,:], 1, np.hstack((self.inputBuffer[n,:], xi[n,:]))) for n in range(self.M)])
        # Save tail end for next time
        self.inputBuffer = xi[:,-self.inputBuffer.shape[1]:]
        # take second part
        xf = xf[:,self.inputBuffer.shape[1]:]

        # App
        xd = np.asarray([np.append(xf[i],0) if i==0 else np.insert(xf[i], 0, 0) for i in range(self.M)])
        xd[1:,0:1] = self.outputBuffer
        self.outputBuffer = xd[1:,-1:]

        return xd[:,:-1]

    def process(self, data):
        # if data.size % self.M != 0:
        #     raise IndexError('Input length must ba a multiple of M [{}]'.format(self.M))
        y = self.polyphase_downFIR(data)
        # output = np.fft.ifft(y, n=self.M, axis=1) * self.M
        output = np.fft.ifft(y, n=self.M, axis=0) * self.M
        return output

def gen_complex_chirp(fs=44100):
    f0=-fs/2.1
    f1=fs/2.1
    t1 = 1
    beta = (f1-f0)/float(t1)
    t = np.arange(0,t1,t1/float(fs))
    return np.exp(2j*np.pi*(.5*beta*(t**2) + f0*t))


if __name__ == "__main__":
    Fs = 44100
    # data = np.ones((4000,), dtype=np.csingle)
    # t = np.arange(0, 4, step=1/rx.sampleRateHz)
    # fc = np.linspace(-1*rx.sampleRateHz/2, 0, t.size)
    # data = np.exp(2j*np.pi*t*fc)

    data = gen_complex_chirp()
    data += .01*np.random.randn(len(data))
    # data = np.arange(data.size)

    # data = np.arange(0,rx.M*50, dtype=np.csingle)
    num_blocks = 10
    block_len = np.floor(data.size/3/num_blocks)
    inds = np.arange(0, block_len*3*10, dtype=int).reshape(num_blocks,-1)

    channelHz = 44100/3
    rx = PolyphaseRxChannelizer(sampleRateHz=channelHz*3, channelBandwidthHz=channelHz, blockSize=block_len)
    print(rx)

    for m in range(num_blocks):
        output = rx.process(data[inds[m,:]])
        
        # print(output.shape)
        # np.set_printoptions(precision=4, linewidth=180)
        # print(output)

        t = np.arange(0, output.shape[1]) + m*output.shape[1]
        for n in range(0, rx.M):
            plt.subplot(rx.M, 1, n+1)
            plt.plot(t, output[n,:])
            plt.grid(True)
    plt.show()
