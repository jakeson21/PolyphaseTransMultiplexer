#!/usr/bin/env python3
import numpy as np
from scipy import signal
import json
from .utils import make_polyphase_filter


class PolyphaseDownFir:
    def __init__(self, channels, h):
        self.M = channels
        self.filter = h
        # inputBuffer is used to save filterlength-1 WB samples for overlap-and-save based filtering
        self.input_buffer = np.zeros((self.M, self.filter.shape[1] - 1), dtype=np.cdouble)
        # outputBuffer stores the last M-1 filtered and reordered samples for use in the next iteration
        self.output_buffer = np.zeros((self.M - 1, 1), dtype=np.cdouble)

    def process(self, x):
        # reshape input in prep for polyphase filtering
        head_data_stream = np.asarray(x[::self.M])
        tail_data_streams = np.asarray([np.asarray(x[i::self.M]) for i in range(1, self.M)])
        xi = np.vstack((head_data_stream, tail_data_streams[::-1, :]))

        # Overlap and save here
        xf = np.asarray([signal.lfilter(self.filter[n, :], 1, np.hstack((self.input_buffer[n, :], xi[n, :]))) for n in range(self.M)])
        # Save tail end for next time
        self.input_buffer = xi[:, -self.input_buffer.shape[1]:]
        # take second part
        xf = xf[:, self.input_buffer.shape[1]:]

        # App
        xd = np.asarray([np.append(xf[i], 0) if i == 0 else np.insert(xf[i], 0, 0) for i in range(self.M)])
        xd[1:, 0:1] = self.output_buffer
        self.output_buffer = xd[1:, -1:]
        return xd[:, :-1]


class PolyphaseUpFir:
    def __init__(self, channels, h):
        self.M = channels
        self.filter = h
        # inputBuffer is used to save filterlength-1 WB samples for overlap-and-save based filtering
        self.input_buffer = np.zeros((self.M, self.filter.shape[1] - 1), dtype=np.cdouble)

    def process(self, x):
        # Filter by Overlap and save
        xf = np.asarray([signal.lfilter(self.filter[n, :], 1, np.hstack((self.input_buffer[n, :], x[n, :]))) for n in range(self.M)])
        # Save tail end for next time
        self.input_buffer = x[:, -self.input_buffer.shape[1]:]
        # take second part
        xf = xf[:, self.input_buffer.shape[1]:]
        # Up-sample: Commute output of filter stages at rate M by taking an output from every filter for each iteration
        return xf.transpose().reshape(-1, )


# Mux
class Synthesis:
    def __init__(self):
        self.M = 2
        self.filter = make_polyphase_filter(self.M, order=40, cutoff=0.5, transition_width=0.2)
        self.up_fir = PolyphaseUpFir(self.M, self.filter)
        self.t_last = 0

    def process(self, x):
        y = np.fft.ifft(x, n=self.M, axis=0) * self.M
        output = self.up_fir.process(y)
        if self.M % 2 == 0:
            t = np.linspace(0, output.size, output.size, endpoint=True) + self.t_last
            self.t_last = t[-1] + t[1]-t[0]
            d = output * np.exp(-2j * np.pi / (2 * self.M) * t)
        return d


# De-mux
class Analysis:
    def __init__(self):
        self.M = 2
        self.filter = make_polyphase_filter(self.M, order=40, cutoff=0.5, transition_width=0.2)
        self.down_fir = PolyphaseDownFir(self.M, self.filter)
        self.t_last = 0

    def process(self, x):
        if self.M % 2 == 0:
            t = np.linspace(0, x.size, x.size, endpoint=True) + self.t_last
            self.t_last = t[-1] + t[1]-t[0]
            d = x * np.exp(-2j * np.pi / (2 * self.M) * t)
        y = self.down_fir.process(d)
        output = np.fft.ifft(y, n=self.M, axis=0) # * self.M
        return output


class OctaveSynthesis:
    def __init__(self, n=1):
        assert n > 0, 'n should be > 0'
        self.num_octaves = n
        self.octaves = {i: {'handle': Synthesis()} for i in range(0, self.num_octaves)}

    def octave_synthesis(self, x, n, y):
        if n > 0:
            # start with last 2 octaves, combine them
            z = np.vstack((x[str(n-1)], y))
            # process 2
            y = self.octaves[n-1]['handle'].process(z)
            n -= 1
            y = self.octave_synthesis(x, n, y)
        return y

    def process(self, x: dict):
        # process data from each octave
        # start by grabbing last element, the smallest one
        y = x[str(self.num_octaves)]
        y = self.octave_synthesis(x, self.num_octaves, y)
        return y


class OctaveAnalysis:
    def __init__(self, n=1):
        assert n > 0, 'n should be > 0'
        self.num_octaves = n
        self.octaves = {str(i): {'handle': Analysis(), 'data': None} for i in range(0, self.num_octaves+1)}

    def octave_analysis(self, x, n=0):
        if n < self.num_octaves-1:
            y = self.octaves[str(n)]['handle'].process(x)
            # Store negative frequency
            self.octaves[str(n)]['data'] = y[1, :]
            # Process positive frequency
            n += 1
            self.octave_analysis(y[0, :], n)
        else:
            # Process final octave
            y = self.octaves[str(n)]['handle'].process(x)
            self.octaves[str(n)]['data'] = y[1, :]
            self.octaves[str(n+1)]['data'] = y[0, :]

    def process(self, x):
        self.octave_analysis(x)
        # Collect data from each octave
        y = dict()
        for k in self.octaves:
            y[k] = self.octaves[k]['data']
        return y

