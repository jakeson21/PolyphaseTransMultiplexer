import numpy as np
from scipy import signal
from transmux.wola_processor import WolaProcessor
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from transmux.utils import gen_real_chirp, gen_fdma, gen_complex_awgn, plot_response
import wave
import struct


num_channels = int(32)
sampleRate = 44100 # hertz


# equalizer
def process_func(x: np.ndarray):
    f = np.fft.rfftfreq(2*(x.size-1), d=1./2./(x.size-1))
    s = 1
    x[f <= s] = x[f <= s] / 2.
    x[f > s] = x[f > s] * 4.
    x[f > 2] = x[f > 2] / 8.
    return x


# data = np.ones((1000,), dtype=np.complex64)
# data = np.arange(int(N/2), dtype=np.complex64)
# data = np.exp(1j*2*np.pi*np.arange(int(100*N))*0.1/N)

# Pulse
# data = np.ones_like(data)
# data[int(-data.size/2):] = 0

# data = gen_complex_awgn(data.size)
# data = gen_real_chirp(fs=sampleRate)
# data = np.fft.ifft(1000*data)

# plot_response(wp.window)


duration = 5.0 # seconds
obj = wave.open('sound.wav', 'w')
obj.setnchannels(1) # mono
obj.setsampwidth(2)
obj.setframerate(sampleRate)
for i in range(int(duration*sampleRate)):
    value = int(np.random.normal(0, 6000))
    samp = struct.pack('<h', value)
    obj.writeframesraw(samp)
obj.close()
obj = wave.open('sound.wav', 'r')
buf = obj.readframes(int(3*sampleRate))
dt = np.dtype(np.int16)
dt = dt.newbyteorder('>')
data = np.frombuffer(buf, dtype=dt)
obj.close()


wp = WolaProcessor(fun=process_func, hop_size=int(num_channels/2), block_size=num_channels, c_valued=False)
y = np.ndarray((0,), dtype=np.float64)

samps_per_step = wp.R
for n in range(0, int((data.size-wp.N)/samps_per_step)):
    inds = np.arange(wp.N) + n*samps_per_step
    d = wp.process(data[inds])
    y = np.hstack((y, d))
    # print(n, d)
print(y)

y_norm = y/np.max(y)*16384.
# Play the sound file
obj = wave.open('sound_out.wav', 'w')
obj.setnchannels(1) # mono
obj.setsampwidth(2)
obj.setframerate(sampleRate)
for i in range(int(y_norm.size)):
    value = int(y_norm[i])
    samp = struct.pack('<h', value)
    obj.writeframesraw(samp)
obj.close()
# os.system('"c:/Program Files (x86)/Windows Media Player/wmplayer.exe" "c:\\git\\PolyphaseTransMultiplexer-develop\\ssound_out.wav"')


fig = plt.figure()
gs = gridspec.GridSpec(4, 1)
ax = fig.add_subplot(gs[0, 0])
ax.plot(data.real, alpha=0.75)
ax.plot(y.real, alpha=0.75)
ax.legend(('x.real', 'xf.real'), loc='upper right')
ax.set_title('Recombined output')
ax.grid(True)
ax = fig.add_subplot(gs[1, 0])
ax.plot(data.imag, alpha=0.75)
ax.plot(y.imag, alpha=0.75)
ax.legend(('x.imag', 'xf.imag'), loc='upper right')
ax.set_title('Recombined output')
ax.grid(True)

ax = fig.add_subplot(gs[2, 0])
ax.specgram(y, NFFT=64, Fs=sampleRate, noverlap=32)
ax.set_title('Output waterfall')
ax.grid(True)
ax = fig.add_subplot(gs[3, 0])
f, Pxx_spec = signal.welch(y, sampleRate, 'flattop', 128, return_onesided=True, scaling='spectrum')
# ax.plot(np.fft.fftshift(f), 20*np.log10(np.abs(np.fft.fftshift(Pxx_spec))))
ax.loglog(f, np.abs(Pxx_spec)**2.)
ax.set_xlabel('frequency [Hz]')
ax.legend(('Original', 'Recombined'), loc='upper right')
ax.set_title('Output spectrum')
ax.grid(True)
plt.show()

