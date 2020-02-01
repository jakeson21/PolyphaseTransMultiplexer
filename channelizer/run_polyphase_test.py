import numpy as np
from scipy import signal
from time import perf_counter
import matplotlib.pyplot as plt
import sys
from polyphase_rxchannelizer import PolyphaseRxChannelizer
from polyphase_txmultiplexer import PolyphaseTxMultiplexer


def rcosdesign(beta, span, sps, name='normal'):
    delay = span * sps / 2
    t = np.arange(-delay, delay + 1) / sps
    if name == 'normal':
        # Design a normal raised cosine filter
        b = np.zeros((t.size,))
        # Find non-zero denominator indices
        denom = 1 - (2 * beta * t) ** 2
        idx1 = np.abs(denom) > np.sqrt(sys.float_info.epsilon)

        # Calculate filter response for non-zero denominator indices
        b[idx1] = np.sinc(t[idx1]) * (np.cos(np.pi * beta * t[idx1]) / denom[idx1]) / sps

        # fill in the zeros denominator indices
        b[idx1 == False] = beta * np.sin(np.pi / (2 * beta)) / (2 * sps)
    else:
        # Design a square root raised cosine filter
        b = np.zeros((t.size,))
        # Find mid-point
        idx1 = t == 0
        if np.any(idx1):
            b[idx1] = -1 / (np.pi * sps) * (np.pi * (beta - 1) - 4 * beta)

        # Find non-zero denominator indices
        idx2 = np.abs(np.abs(4. * beta * t) - 1.0) < np.sqrt(sys.float_info.epsilon)
        if np.any(idx2):
            b[idx2] = 1 / (2 * np.pi * sps) \
                      * (np.pi * (beta + 1) * np.sin(np.pi * (beta + 1) / (4 * beta))
                         - 4 * beta * np.sin(np.pi * (beta - 1) / (4 * beta))
                         + np.pi * (beta - 1) * np.cos(np.pi * (beta - 1) / (4 * beta)))

        # fill in the zeros denominator indices
        nind = t[idx1 + idx2 == False]

        b[idx1 + idx2 == False] = (-4 * beta / sps) * (np.cos((1 + beta) * np.pi * nind) + np.sin((1 - beta) * np.pi * nind) / (4 * beta * nind)) / (np.pi * ((4 * beta * nind) ** 2 - 1))

    # Normalize filter energy
    b = b / np.sqrt(np.sum(b ** 2))
    return b


def gen_complex_chirp(fs=44100):
    f0 = -fs / 2.1
    f1 = fs / 2.1
    t1 = 1
    beta = (f1 - f0) / float(t1)
    t = np.arange(0, t1, t1 / float(fs))
    return np.exp(2j * np.pi * (.5 * beta * (t ** 2) + f0 * t))


def gen_fdma(fs, bw):
    num_chans = int(np.floor(fs / bw))
    # Generate QPSK data of about 1 second
    C = np.asarray([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])
    S = C[np.random.randint(C.size, size=(bw, 1))]
    Sup = np.zeros((S.size, num_chans - 1,))
    x = np.hstack((S, Sup))
    data = np.squeeze(np.reshape(x, (x.size, -1)))
    hrrc = np.asarray(rcosdesign(beta=0.01, span=256, sps=num_chans * 2, name='normal'), dtype=np.cdouble)
    data = signal.lfilter(hrrc, 1, data)

    # generate fdma mapping
    f = np.linspace(-fs / 2, fs / 2, num_chans * 2 + 1)[1::2]
    fdma = np.random.randint(num_chans, size=10 * num_chans)
    bs = int(np.floor(data.size / fdma.size))

    n = 0
    t = np.arange(0, data.size)
    for m in t[::bs]:
        fc = f[fdma[n]]
        z = np.arange(0, bs, dtype=int) + m
        p = np.ones_like(np.asarray(data[z]))
        L = int(np.round(0.05 * bs))
        p[:L] = np.logspace(-6, 0, num=L)
        p[-L:] = np.logspace(0, -6, num=L)
        data[z] *= np.exp(2j * np.pi * fc * t[z] / fs) * p
        n += 1
    # Add noise to FDMA signal
    data += 0.01 * np.squeeze(np.random.randn(data.size, 2).view(np.complex128))
    return data


if __name__ == "__main__":
    num_channels = 5
    chanBW = 5000
    Fs = chanBW * num_channels
    data = gen_fdma(fs=Fs, bw=chanBW)

    # data = np.arange(0,rx.M*50, dtype=np.cdouble)
    num_blocks = 10
    block_len = int(np.floor(data.size / num_channels / num_blocks))
    inds = np.arange(0, block_len * num_channels * num_blocks, dtype=int).reshape(num_blocks, -1)

    channelHz = Fs / num_channels
    rx = PolyphaseRxChannelizer(sample_rate_Hz=Fs, channel_bandwidth_Hz=channelHz, block_size=block_len)
    print(rx)

    # Start the stopwatch / counter
    t1_start = perf_counter()
    for m in range(num_blocks):
        output = rx.process(data[inds[m, :]])
        # print(output.shape)
        # np.set_printoptions(precision=4, linewidth=180)
        # print(output)
        if m == 0:
            nb_outputs = output
        else:
            nb_outputs = np.hstack((nb_outputs, output))
    # Stop the stopwatch / counter
    t1_stop = perf_counter()
    print("Elapsed time: {} s".format(t1_stop - t1_start))
    print("Samples per second: {}".format(num_blocks*output.size / (t1_stop - t1_start)))

    tx = PolyphaseTxMultiplexer(sample_rate_Hz=Fs, channel_bandwidth_Hz=channelHz, block_size=block_len)
    inds = np.arange(0, block_len * num_blocks, dtype=int).reshape(num_blocks, -1)
    for m in range(num_blocks):
        output = tx.process(nb_outputs[:, inds[m, :]])
        if m == 0:
            wb_output = output
        else:
            wb_output = np.hstack((wb_output, output))
    wb_output = wb_output[tx.input_buffer.size:]

    # Channelizer plot
    plt.rcParams.update({'font.size': 7})
    fig, ax = plt.subplots(num_channels + 2, 2)
    t = np.arange(0, nb_outputs.shape[1]) / Fs * num_channels
    for n in range(0, rx.M):
        ax[n + 1, 0].specgram(nb_outputs[n, :], NFFT=32, Fs=Fs / num_channels, noverlap=16)
        ax[n + 1, 1].plot(t, 20 * np.log10(np.abs(nb_outputs[n, :])))
        ax[n + 1, 0].grid(True)
        ax[n + 1, 1].grid(True)
        ax[n + 1, 1].set_ylim(top=20)
        ax[n + 1, 1].set_ylim(bottom=-60)
    # Channelizer input plot
    t = np.arange(0, data.size) / Fs
    ax[0, 0].specgram(data, NFFT=128, Fs=Fs, noverlap=100)
    ax[0, 1].plot(t, 20 * np.log10(np.abs(data)))
    ax[0, 0].grid(True)
    ax[0, 1].grid(True)
    ax[0, 1].set_ylim(top=20)
    ax[0, 1].set_ylim(bottom=-60)
    # Multiplexer output plot
    t = np.arange(0, wb_output.size) / Fs
    ax[num_channels + 1, 0].specgram(wb_output, NFFT=128, Fs=Fs, noverlap=100)
    ax[num_channels + 1, 1].plot(t, 20 * np.log10(np.abs(wb_output)))
    ax[num_channels + 1, 0].grid(True)
    ax[num_channels + 1, 1].grid(True)
    ax[num_channels + 1, 1].set_ylim(top=20)
    ax[num_channels + 1, 1].set_ylim(bottom=-60)
    plt.show()


