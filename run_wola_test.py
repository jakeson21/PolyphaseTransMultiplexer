import numpy as np
from scipy import signal
from transmux.wola_oversampled_channelizer import WolaChannelizer, plot_response
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from transmux.utils import gen_complex_chirp, gen_fdma, gen_complex_awgn


fs = 1000
N = 10
rx = WolaChannelizer(sample_rate_hz=fs, channels=N)

# data = np.ones(int(N/2,), dtype=np.complex64)
# data = np.arange(int(N/2), dtype=np.complex64)
data = np.exp(1j*2*np.pi*np.arange(int(100*N))*0.1/N)

# data = 1000*np.ones_like(data)
# data[int(-data.size/2):] = 0

data = gen_complex_awgn(data.size)

# print(data)
y = np.ndarray((0,), dtype=np.complex128)
for n in range(0, int(data.size/rx.R)):
    d = rx.process(data[n*rx.R:n*rx.R+rx.R])
    y = np.hstack((y, d))
    # print(n, d)


fig = plt.figure()
gs = gridspec.GridSpec(2, 1)
ax = fig.add_subplot(gs[0, 0])
ax.plot(y.real, alpha=1.)
ax.plot(y.imag, alpha=1.)
ax.grid(True)

ax = fig.add_subplot(gs[1, 0])
win = rx.window
for n in range(1, 8):
    win += np.roll(rx.window, n*rx.R)
ax.plot(win)
ax.grid(True)
plt.show()

fig = plt.figure()
gs = gridspec.GridSpec(2, 1)
ax = fig.add_subplot(gs[0, 0])
ax.specgram(y, NFFT=64, Fs=fs, noverlap=32)
ax.grid(True)
ax = fig.add_subplot(gs[1, 0])
f, Pxx_spec = signal.welch(y, fs, 'flattop', 64, return_onesided=False, scaling='spectrum')
ax.plot(np.fft.fftshift(f), 20*np.log10(np.abs(np.fft.fftshift(Pxx_spec))))
ax.set_xlabel('frequency [Hz]')
ax.legend(('Original', 'Recombined'), loc='upper right')
ax.grid(True)
plt.show()

