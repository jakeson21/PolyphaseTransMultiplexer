#!/usr/bin/env python3
from typing import Mapping, MutableMapping, Sequence, Iterable, List, Set, NewType, Dict
import numpy as np
from scipy import signal
import uuid
import json

def MakeWindow(length: int = 0):
    desired = (1, 1, 0, 0)
    bands = (0, 0.5, 0.5, 1)
    w = signal.firls(length+1, bands, desired)
    w = w[0:-1]
    return w


class OutputTuner():
    def __init__(self, nbSampleRateHz: int, fc: float=0):
        self.nbSampleRateHz = nbSampleRateHz
        self.fc = fc


TunerType = NewType('TunerType', OutputTuner)
TunerMapping = Dict[int, TunerType]


class RxChannelizer():

    def __init__(self, wbSampleRateHz: float, wbStepSizeHz: int):
        if wbSampleRateHz % wbSampleRateHz != 0 and wbSampleRateHz/wbSampleRateHz % 2 != 0:
            raise ValueError('wbSampleRateHz/wbSampleRateHz {}/{} must be an integer and even'.format(wbSampleRateHz,wbSampleRateHz))
        self.wbSampleRateHz = wbSampleRateHz
        self.wbStepSizeHz = wbStepSizeHz

        # Constant algorithm parameters
        self.oversample: int = 4
        self.overlap: float = 0.5

        # Derived algorithm parameters
        self.nfft: int = self.wbSampleRateHz/self.wbStepSizeHz
        self.R: int = int(self.nfft*self.overlap)

        # Algorithm Buffers
        self.inputBuffer = np.ndarray((int(self.nfft*self.oversample),), dtype=np.csingle)
        self.window = MakeWindow(int(self.nfft*self.oversample))

        # Workload variables
        self.Tuners = dict()

    def __str__(self):
        d = dict()
        d['wbSampleRateHz'] = self.wbSampleRateHz
        d['wbStepSizeHz'] = self.wbStepSizeHz
        d['oversample'] = self.oversample
        d['overlap'] = self.overlap
        d['nfft'] = self.nfft
        d['R'] = self.R
        d['window_len'] = self.window.size
        d['buffer_len'] = self.inputBuffer.size
        return json.dumps(d)

    def requestTuner(self, nbSampleRateHz: int, fc: float=0):
        tuner_id = uuid.uuid1()
        self.Tuners[tuner_id] = OutputTuner(nbSampleRateHz, fc)
        return self.Tuners[tuner_id]

    def numTuners(self) -> int:
        return len(self.Tuners)

    def process(self):
        # Do algorithm here
        
        pass





if __name__ == "__main__":
    rxchan = RxChannelizer(1000, 100)
    print (rxchan)
    t1 = rxchan.requestTuner(800, 300)
    print(t1)
    print(rxchan.numTuners())
