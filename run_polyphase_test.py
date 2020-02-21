import numpy as np
from scipy import signal
from time import perf_counter
import matplotlib.pyplot as plt
import sys

from transmux.polyphase_rxchannelizer import PolyphaseRxChannelizer
from transmux.polyphase_txmultiplexer import PolyphaseTxMultiplexer

from transmux.halfband import Synthesis, Analysis, OctaveAnalysis
from transmux.utils import gen_complex_chirp, gen_fdma


if __name__ == "__main__":
    num_channels = 5
    chanBW = 100000
    Fs = chanBW * num_channels
    # data = gen_fdma(fs=Fs, bw=chanBW)
    data = gen_complex_chirp(fs=Fs, duration=2)

    # data = np.arange(0,rx.M*50, dtype=np.cdouble)
    num_blocks = 10
    block_len = int(np.floor(data.size / num_channels / num_blocks))
    inds = np.arange(0, block_len * num_channels * num_blocks, dtype=int).reshape(num_blocks, -1)

    channelHz = Fs / num_channels
    rx = PolyphaseRxChannelizer(sample_rate_Hz=Fs, channel_bandwidth_Hz=channelHz)
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

    tx = PolyphaseTxMultiplexer(sample_rate_Hz=Fs, channel_bandwidth_Hz=channelHz)
    print(tx)
    inds = np.arange(0, block_len * num_blocks, dtype=int).reshape(num_blocks, -1)
    t1_start = perf_counter()
    for m in range(num_blocks):
        output = tx.process(nb_outputs[:, inds[m, :]])
        if m == 0:
            wb_output = output
        else:
            wb_output = np.hstack((wb_output, output))
    # Stop the stopwatch / counter
    t1_stop = perf_counter()
    print("Elapsed time: {} s".format(t1_stop - t1_start))
    print("Samples per second: {}".format(num_blocks*output.size / (t1_stop - t1_start)))

    # wb_output = wb_output[tx.input_buffer.size:]

    # Channelizer plot
    plt.rcParams.update({'font.size': 7})
    fig, ax = plt.subplots(num_channels + 2, 2)
    t = np.arange(0, nb_outputs.shape[1]) / Fs * num_channels
    for n in range(0, rx.M):
        ax[n + 1, 0].specgram(nb_outputs[n, :], NFFT=128, Fs=Fs / num_channels, noverlap=100)
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

    f, Pxx_spec = signal.welch(data, Fs, 'flattop', 1024, return_onesided=False, scaling='spectrum')
    plt.figure()
    plt.semilogy(np.fft.fftshift(f), np.fft.fftshift(Pxx_spec))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Linear spectrum [mW RMS]')
    plt.show()
