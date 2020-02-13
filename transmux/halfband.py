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
        xf = np.asarray(
            [signal.lfilter(self.filter[n, :], 1, np.hstack((self.input_buffer[n, :], xi[n, :]))) for n in range(self.M)])
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
    def __init__(self, handle=None):
        self.M = 2
        self.filter = make_polyphase_filter(self.M)
        self.up_fir = PolyphaseUpFir(self.M, self.filter)
        self.t_last = 0
        self.handle = handle

    def process(self, x):
        # if self.handle is not None:
        #     x = self.handle.process(x)
        y = np.fft.ifft(x, n=self.M, axis=0) * self.M
        output = self.up_fir.process(y)
        if self.M % 2 == 0:
            t = np.linspace(0, output.size, output.size, endpoint=True) + self.t_last
            self.t_last = t[-1] + t[1]-t[0]
            output *= np.exp(2j * np.pi / (2 * self.M) * t)
        return output


# De-mux
class Analysis:
    def __init__(self, handle=None):
        self.M = 2
        self.filter = make_polyphase_filter(self.M)
        self.down_fir = PolyphaseDownFir(self.M, self.filter)
        self.t_last = 0
        self.handle = handle

    def process(self, x):
        # if self.handle is not None:
        #     x = self.handle.process(x)
        if self.M % 2 == 0:
            t = np.linspace(0, x.size, x.size, endpoint=True) + self.t_last
            self.t_last = t[-1] + t[1]-t[0]
            x *= np.exp(-2j * np.pi / (2 * self.M) * t)
        y = self.down_fir.process(x)
        output = np.fft.ifft(y, n=self.M, axis=0) * self.M
        return output
