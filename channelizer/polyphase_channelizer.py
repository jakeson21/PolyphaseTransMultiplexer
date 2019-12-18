#!/usr/bin/env python3
from typing import Mapping, MutableMapping, Sequence, Iterable, List, Set, NewType, Dict
import numpy as np
from scipy import signal
import json
import pdb
import matplotlib.pyplot as plt
import matplotlib
import sys
from time import perf_counter

class PolyphaseRxChannelizer():
    def __init__(self, sampleRateHz: float, channelBandwidthHz: float):
        self.sampleRateHz: float = sampleRateHz
        self.channelBandwidthHz: float = channelBandwidthHz

        # Ensure total bandwidth is evenly divisible by the channel bandwidth
        if (sampleRateHz/channelBandwidthHz) != np.round(sampleRateHz/channelBandwidthHz):
            raise ValueError('sampleRateHz ({}) must be evenly divisible by channelBandwidthHz ({})'.format(sampleRateHz, channelBandwidthHz))

        self.M = int(self.sampleRateHz / self.channelBandwidthHz)
        Order = 10
        self.filter = self.makePolyphaseFilter(order=Order)
        # inputBuffer is used to save filterlength-1 WB samples for overlap-and-save based filtering
        self.inputBuffer = np.zeros((self.M, self.filter.shape[1]-1), dtype=np.cdouble)
        # outputBuffer stores the last M-1 filtered and reordered samples for use in the next iteration
        self.outputBuffer = np.zeros((self.M-1, 1), dtype=np.cdouble)

    def __str__(self):
        d = dict()
        d['sampleRateHz'] = self.sampleRateHz
        d['channelBandwidthHz'] = self.channelBandwidthHz
        d['M'] = self.M
        d['filter'] = self.filter.tolist()
        return json.dumps(d)

    def makePolyphaseFilter(self, order):
        fs = self.M
        # cutoff = 1.0/(self.M - 0.25*self.M)
        cutoff = 0.5
        transition_width = 0.1
        numtaps, beta = signal.kaiserord(150, width=transition_width/(0.5*fs))
        print('fs={}, transition_width={}, cutoff={}, numtaps={}, beta={}'.format(fs, transition_width, cutoff, numtaps, beta))

        h = signal.firwin(numtaps, cutoff=cutoff, window=('kaiser', beta), scale=False, nyq=0.5*fs)
        # h[abs(h) <= 1e-4] = 0.
        # h = h/np.max(h)/fs

        # form Hk
        N: int = int(np.ceil(np.ceil(h.size/fs)*fs))
        h.resize((N,), refcheck=False)
        # Columns are filter paths
        Hk = np.reshape(h,(-1, fs)).transpose()
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


def rcosdesign(beta, span, sps, name='normal'):
    delay = span*sps/2
    t = np.arange(-delay, delay+1)/sps
    if name == 'normal':
        # Design a normal raised cosine filter
        b = np.zeros((t.size,))
        # Find non-zero denominator indices
        denom = 1-(2*beta*t)**2
        idx1 = np.abs(denom) > np.sqrt(sys.float_info.epsilon)
        
        # Calculate filter response for non-zero denominator indices
        b[idx1] = np.sinc(t[idx1])*(np.cos(np.pi*beta*t[idx1])/denom[idx1])/sps
        
        # fill in the zeros denominator indices
        b[idx1==False] = beta * np.sin(np.pi/(2*beta)) / (2*sps)
    else:
        # Design a square root raised cosine filter
        b = np.zeros((t.size,))
        # Find mid-point
        idx1 = t == 0
        if np.any(idx1):
            b[idx1] = -1 / (np.pi*sps) * (np.pi * (beta-1) - 4*beta)
        
        # Find non-zero denominator indices
        idx2 = np.abs(np.abs(4.*beta*t) - 1.0) < np.sqrt(sys.float_info.epsilon)
        if np.any(idx2):
            b[idx2] = 1 / (2*np.pi*sps) \
            * (np.pi * (beta+1) * np.sin(np.pi*(beta+1)/(4*beta)) \
            - 4*beta * np.sin(np.pi*(beta-1)/(4*beta)) \
            + np.pi*(beta-1) * np.cos(np.pi*(beta-1)/(4*beta)))
        
        # fill in the zeros denominator indices
        nind = t[idx1+idx2==False]
        
        b[idx1+idx2==False] = (-4*beta/sps) \
            * (np.cos((1+beta)*np.pi*nind) + np.sin((1-beta)*np.pi*nind) / (4*beta*nind) ) \
            / (np.pi * ((4*beta*nind)**2 - 1))
        
    # Normalize filter energy
    b = b / np.sqrt(np.sum(b**2))
    return b

def gen_complex_chirp(fs=44100):
    f0=-fs/2.1
    f1=fs/2.1
    t1 = 1
    beta = (f1-f0)/float(t1)
    t = np.arange(0,t1,t1/float(fs))
    return np.exp(2j*np.pi*(.5*beta*(t**2) + f0*t))

def gen_fdma(fs, bw):
    num_chans = int(np.floor(fs/bw))
    # Generate QPSK data of about 1 second
    C = np.asarray([1+1j, 1-1j, -1+1j, -1-1j])
    S = C[np.random.randint(C.size, size=(bw, 1))]
    Sup = np.zeros((S.size, num_chans-1,))
    x = np.hstack((S,Sup))
    data = np.squeeze(np.reshape(x, (x.size,-1)))
    hrrc = np.asarray(rcosdesign(beta=0.01, span=256, sps=num_chans*2, name='sqrt'), dtype=np.cdouble)
    data = signal.lfilter(hrrc, 1, data)

    # generate fdma mapping
    f = np.linspace(-fs/2, fs/2, num_chans*2+1)[1::2]
    fdma = np.random.randint(num_chans, size=10*num_chans)
    bs = int(np.floor(data.size/fdma.size))

    n = 0
    t = np.arange(0,data.size)
    for m in t[::bs]:
        fc = f[fdma[n]]
        z = np.arange(0, bs, dtype=int) + m
        p = np.ones_like(np.asarray(data[z]))
        L = int(np.round(0.05*bs))
        p[:L] = np.logspace(-6, 0, num=L)
        p[-L:] = np.logspace(0, -6, num=L)
        data[z] *= np.exp(2j*np.pi*fc*t[z]/fs) * p
        n += 1

    return data

if __name__ == "__main__":
    num_channels = 15
    chanBW = 50000
    Fs = chanBW*num_channels
    # data = np.ones((4000,), dtype=np.cdouble)
    # t = np.arange(0, 4, step=1/rx.sampleRateHz)
    # fc = np.linspace(-1*rx.sampleRateHz/2, 0, t.size)
    # data = np.exp(2j*np.pi*t*fc)

    # data = gen_complex_chirp(fs=Fs)
    # data += .0001*np.random.randn(len(data))
    data = gen_fdma(fs=Fs, bw=chanBW)

    # data = np.arange(0,rx.M*50, dtype=np.cdouble)
    num_blocks = 10
    block_len = np.floor(data.size/num_channels/num_blocks)
    inds = np.arange(0, block_len*num_channels*num_blocks, dtype=int).reshape(num_blocks,-1)

    channelHz = Fs/num_channels
    rx = PolyphaseRxChannelizer(sampleRateHz=Fs, channelBandwidthHz=channelHz)
    print(rx)

    # Start the stopwatch / counter 
    t1_start = perf_counter()
    for m in range(num_blocks):
        output = rx.process(data[inds[m,:]])
        # print(output.shape)
        # np.set_printoptions(precision=4, linewidth=180)
        # print(output)
        if m == 0:
            outputs = output
        else:
            outputs = np.hstack((outputs, output))
    # Stop the stopwatch / counter 
    t1_stop = perf_counter()
    print("Elapsed time: {} s".format(t1_stop-t1_start))
    print("Samples per second: {}".format(outputs.size/(t1_stop-t1_start)))

    # Make pretty plots
    # plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 7})
    fig, ax = plt.subplots(num_channels+1, 2)
    t = np.arange(0, outputs.shape[1])/Fs*num_channels
    for n in range(0, rx.M):
        ax[n+1,0].specgram(outputs[n,:], NFFT=32, Fs=Fs/num_channels, noverlap=16)
        ax[n+1,1].plot(t, 20*np.log10(np.abs(outputs[n,:])))
        ax[n+1,0].grid(True)
        ax[n+1,1].grid(True)
        ax[n+1,1].set_ylim(top=20)
        ax[n+1,1].set_ylim(bottom=-100)
    t = np.arange(0, data.size)/Fs
    ax[0,0].specgram(data, NFFT=128, Fs=Fs, noverlap=100)
    ax[0,1].plot(t, 20*np.log10(np.abs(data)))
    ax[0,0].grid(True)
    ax[0,1].grid(True)
    ax[0,1].set_ylim(top=20)
    ax[0,1].set_ylim(bottom=-100)
    plt.show()
