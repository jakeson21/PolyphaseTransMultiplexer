#!/usr/bin/env python3
from typing import Mapping, MutableMapping, Sequence, Iterable, List, Set, NewType, Dict
import numpy as np
from scipy import signal
import uuid
import json
import pdb
import matplotlib.pyplot as plt

def MakeWindow(nfft: int = 0, oversample: int = 4):
    f = 1.237 / nfft
    desired = (1, 1, 0, 0)
    bands = (0, f, f, 1)
    lp = signal.firls(nfft*oversample+1, bands, desired)
    w = lp * signal.windows.kaiser(oversample*nfft+1, 7.3)
    w = w[0:-1]
    w = w / np.sum(w)
    return w

def getWbIndices(fcHz: float, stepSizeHz: int, nfft: int, bandwidthHz: int):
    num_bins = np.round(bandwidthHz/stepSizeHz)
    if num_bins % 2 != 0:
        raise ValueError('The number of output bins [{}] must be even'.format(num_bins))
    
    # Center frequency limits
    Fs = nfft*stepSizeHz
    fcHz = fcHz % Fs
    # Calculate center bin
    ind_offset = np.round(fcHz/stepSizeHz)
    f_offset = fcHz - ind_offset*stepSizeHz
    # make indices
    bins = np.arange(-num_bins/2, num_bins/2, dtype=np.int16) + ind_offset
    #  Correct for DFT wrap around
    negwrap = bins<0
    bins[negwrap] = bins[negwrap] + nfft
    poswrap = bins>=nfft
    bins[poswrap] = bins[poswrap] - nfft
    print(bins.shape)
    bins = np.fft.ifftshift(bins)
    bins = bins.astype(int)
    return (bins, -f_offset)


class OutputTuner():
    def __init__(self, tuner_id: int, wb_nfft: int, stepSizeHz: int, bandwidthHz: int, fcHz: float=0):
        if bandwidthHz % stepSizeHz != 0 and bandwidthHz/stepSizeHz % 2 != 0 and bandwidthHz/stepSizeHz > 8:
            raise ValueError('bandwidthHz/wbSampleRateHz [{}/{}] must be an integer and even'.format(bandwidthHz, stepSizeHz))
        self.tuner_id = tuner_id
        self.bandwidthHz = bandwidthHz
        self.fcHz = fcHz

        # Constant algorithm parameters
        self.spread: int = 4
        self.overlap: float = 0.5

        # Derived algorithm parameters
        self.nfft: int = self.bandwidthHz/stepSizeHz
        self.R: int = int(self.nfft*self.overlap)
        self.wb_indices, self.fc_offset = getWbIndices(fcHz=self.fcHz, stepSizeHz=stepSizeHz, nfft=wb_nfft, bandwidthHz=bandwidthHz)
        self.fc_phase_delta = self.fc_offset / bandwidthHz
        self.fc_phase_vec = np.arange(0, self.R)
        self.phase = 0

        # Algorithm Buffers
        self.window = MakeWindow(self.nfft, self.spread) * bandwidthHz**2
        self.outputBuffer = np.zeros((int(self.nfft*self.spread),), dtype=np.csingle)

    def __str__(self):
        d = dict()
        d['tuner_id'] = self.tuner_id
        d['bandwidthHz'] = self.bandwidthHz
        d['spread'] = self.spread
        d['overlap'] = self.overlap
        d['nfft'] = self.nfft
        d['R'] = self.R
        d['window_len'] = self.window.size
        d['buffer_len'] = self.outputBuffer.size
        d['wb_indices'] = self.wb_indices.tolist()
        d['fc_offset'] = self.fc_offset
        d['fc_phase_delta'] = self.fc_phase_delta
        return json.dumps(d)


TunerType = NewType('TunerType', OutputTuner)
TunerMapping = Dict[int, TunerType]


class RxChannelizer():

    def __init__(self, sampleRateHz: int, stepSizeHz: int):
        if sampleRateHz % stepSizeHz != 0 and sampleRateHz/stepSizeHz % 2 != 0:
            raise ValueError('sampleRateHz/stepSizeHz [{}/{}] must be an integer and even'.format(sampleRateHz, stepSizeHz))
        self.sampleRateHz = sampleRateHz
        self.stepSizeHz = stepSizeHz

        # Constant algorithm parameters
        self.oversample: int = 4
        self.overlap: float = 0.5

        # Derived algorithm parameters
        self.nfft: int = self.sampleRateHz/self.stepSizeHz
        self.R: int = int(self.nfft*self.overlap)

        # Algorithm Buffers
        self.window = MakeWindow(self.nfft, self.oversample) * self.R / self.stepSizeHz / self.sampleRateHz
        self.inputBuffer = np.zeros((self.window.size,), dtype=np.csingle)

        # Workload variables
        self.Tuners = dict()

    def __str__(self):
        d = dict()
        d['sampleRateHz'] = self.sampleRateHz
        d['stepSizeHz'] = self.stepSizeHz
        d['oversample'] = self.oversample
        d['overlap'] = self.overlap
        d['nfft'] = self.nfft
        d['R'] = self.R
        d['window_len'] = self.window.size
        d['buffer_len'] = self.inputBuffer.size
        d['Tuners'] = list()
        for key in self.Tuners:
            d['Tuners'].append(json.loads(str(self.Tuners[key])))
        return json.dumps(d)

    def requestTuner(self, bandwidthHz: int, fcHz: float=0):
        tuner_id = uuid.uuid1().hex
        self.Tuners[tuner_id] = OutputTuner(tuner_id=tuner_id, stepSizeHz=self.stepSizeHz, wb_nfft=self.nfft, bandwidthHz=bandwidthHz, fcHz=fcHz)
        return self.Tuners[tuner_id]

    def numTuners(self) -> int:
        return len(self.Tuners)

    def process(self, x):
        # Do algorithm here
        if x.size != self.R:
            raise ValueError('input must be length R {}'.format(self.R))
        # shift buffer left by R
        self.inputBuffer[:self.inputBuffer.size-self.R] = self.inputBuffer[self.R:]
        # place new data in last R
        self.inputBuffer[-self.R:] = x
        # window and sum the buffer
        windowed = self.inputBuffer * self.window
        summed = np.reshape(windowed, (-1, int(self.nfft)))
        summed = np.sum(summed, axis=0)
        X = np.fft.fft(summed)
        output = dict()
        for key in self.Tuners:
            tuner = self.Tuners[key]
            # grab bins representing tuner Fc and bandwidth
            Y = X[tuner.wb_indices]
            if tuner.phase == 1 and tuner.wb_indices[0] % 2 == 1:
                print('flipping phase')
                Y = -Y
            tuner.phase ^= 1
            # to time-domain
            y = np.fft.ifft(Y)
            # spread and window
            ys = np.tile(y, int(tuner.spread))
            yw = ys * tuner.window
            # overlap and add, but shift buffer by R first
            tuner.outputBuffer[:tuner.outputBuffer.size-tuner.R] = tuner.outputBuffer[tuner.R:]
            tuner.outputBuffer[tuner.outputBuffer.size-tuner.R:] = 0
            tuner.outputBuffer += yw
            freq_shift = np.exp(2j*np.pi*tuner.fc_phase_vec*tuner.fc_phase_delta)
            tuner.fc_phase_vec += tuner.R
            output[tuner.tuner_id] = tuner.outputBuffer[:tuner.R] * freq_shift

        return output

if __name__ == "__main__":
    rxchan = RxChannelizer(sampleRateHz=10000, stepSizeHz=100)
    tuners = dict()
    T = rxchan.requestTuner(bandwidthHz=1000, fcHz=0)
    tuners[T.tuner_id] = T
    T = rxchan.requestTuner(bandwidthHz=1000, fcHz=50)
    tuners[T.tuner_id] = T
    T = rxchan.requestTuner(bandwidthHz=2600, fcHz=-500)
    tuners[T.tuner_id] = T

    print(rxchan)

    data = np.ones((rxchan.R, ), dtype=np.csingle)
    # x = np.empty((0,), dtype=np.csingle)
    x = dict()
    plt.subplot(len(tuners)+1, 1, 1)
    for k in range(0,64):
        output = rxchan.process(data)
        # print(output)

        n = 1
        for key in output:
            t = np.arange(0.0, rxchan.Tuners[key].R) + (k-1)*rxchan.Tuners[key].R
            # plt.subplot(len(tuners)+1, 1, 1)
            plt.plot(t/tuners[key].bandwidthHz, output[key].real)
            n += 1
            if key in x:
                x[key] = np.concatenate((x[key], output[key]), axis=0)
            else:
                x[key] = output[key]


    plt.grid(True)
    # plt.show()
    n = 2
    for key in x:
        plt.subplot(len(tuners)+1, 1, n)
        freq = np.fft.fftshift(np.fft.fftfreq(x[key].size))*tuners[key].bandwidthHz
        X_dB = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[key]))))
        # print(X_dB)
        plt.plot(freq, X_dB)
        plt.grid(True)
        n += 1
    plt.show()
