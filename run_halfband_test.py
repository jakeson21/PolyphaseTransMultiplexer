#!/usr/bin/env python3
import numpy as np
from scipy import signal
from time import perf_counter
import matplotlib.pyplot as plt

from transmux.halfband import OctaveAnalysis, OctaveSynthesis
from transmux.utils import gen_complex_chirp, gen_fdma


if __name__ == "__main__":
    num_channels = 10
    chanBW = 100000
    Fs = chanBW * num_channels
    np.random.seed(0)
    # data = gen_fdma(fs=Fs, bw=chanBW)
    data = gen_complex_chirp(fs=Fs, duration=1)

    # data = np.arange(0,rx.M*50, dtype=np.cdouble)
    num_blocks = 10
    block_len = int(np.floor(data.size / num_channels / num_blocks))

    channelHz = Fs / num_channels
    rx = OctaveAnalysis(5)
    print(rx)
    tx = OctaveSynthesis(5)
    print(tx)

    # Start the stopwatch / counter
    t1_start = perf_counter()
    nb_outputs = rx.process(data)

    # Stop the stopwatch / counter
    t1_stop = perf_counter()
    print("Elapsed time: {} s".format(t1_stop - t1_start))
    sample_count = sum(nb_outputs[k].size for k in nb_outputs)
    print("Samples per second: {}".format(num_blocks*sample_count / (t1_stop - t1_start)))

    # inds = np.arange(0, block_len * num_blocks, dtype=int).reshape(num_blocks, -1)
    t1_start = perf_counter()
    wb_output = tx.process(nb_outputs)

    # Stop the stopwatch / counter
    t1_stop = perf_counter()
    print("Elapsed time: {} s".format(t1_stop - t1_start))
    print("Samples per second: {}".format(num_blocks*wb_output.size / (t1_stop - t1_start)))

    # Channelizer plot
    plt.rcParams.update({'font.size': 7})
    fig, ax = plt.subplots(len(nb_outputs) + 2, 2)
    chan_samp_rate = Fs/2
    for n in range(0, len(nb_outputs)):
        t = np.arange(0, nb_outputs[str(n)].shape[0]) / chan_samp_rate
        ax[n + 1, 0].specgram(nb_outputs[str(n)], NFFT=128, Fs=chan_samp_rate, noverlap=100)
        ax[n + 1, 1].plot(t, 20 * np.log10(np.abs(nb_outputs[str(n)])))
        ax[n + 1, 0].grid(True)
        ax[n + 1, 1].grid(True)
        ax[n + 1, 1].set_ylim(top=20)
        ax[n + 1, 1].set_ylim(bottom=-40)
        if n < len(nb_outputs)-2:
            chan_samp_rate = chan_samp_rate / 2

    # Channelizer input plot
    t = np.arange(0, data.size) / Fs
    ax[0, 0].specgram(data, NFFT=128, Fs=Fs, noverlap=100)
    ax[0, 1].plot(t, 20 * np.log10(np.abs(data)))
    ax[0, 0].grid(True)
    ax[0, 1].grid(True)
    ax[0, 1].set_ylim(top=20)
    ax[0, 1].set_ylim(bottom=-40)
    # Multiplexer output plot
    t = np.arange(0, wb_output.size) / Fs
    ax[len(nb_outputs) + 1, 0].specgram(wb_output, NFFT=128, Fs=Fs, noverlap=100)
    ax[len(nb_outputs) + 1, 1].plot(t, 20 * np.log10(np.abs(wb_output)))
    ax[len(nb_outputs) + 1, 0].grid(True)
    ax[len(nb_outputs) + 1, 1].grid(True)
    ax[len(nb_outputs) + 1, 1].set_ylim(top=20)
    ax[len(nb_outputs) + 1, 1].set_ylim(bottom=-40)
    plt.show()

    # t = np.arange(0, nb_outputs.shape[1]) / Fs * num_channels
    # sum_outputs = np.zeros_like(nb_outputs[0, :])
    # fig, ax = plt.subplots(2, 1)
    # for n in range(0, rx.M):
    #     ax[0].plot(t, 20 * np.log10(np.abs(nb_outputs[n, :])))
    #     sum_outputs += nb_outputs[n, :]
    # ax[1].plot(t, 20 * np.log10(np.abs(sum_outputs)))
    # ax[0].grid(True)
    # ax[1].grid(True)
    # ax[0].set_ylim(top=2.5)
    # ax[0].set_ylim(bottom=-2.5)
    # ax[1].set_ylim(top=2.5)
    # ax[1].set_ylim(bottom=-2.5)
    # plt.show()

    # f, Pxx_spec = signal.welch(data, Fs, 'flattop', 1024, return_onesided=False, scaling='spectrum')
    # plt.figure()
    # plt.semilogy(np.fft.fftshift(f), np.fft.fftshift(Pxx_spec))
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('Linear spectrum [mW RMS]')
    # plt.show()
