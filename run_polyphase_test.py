import numpy as np
from scipy import signal
from time import perf_counter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from transmux.polyphase_rxchannelizer import PolyphaseRxChannelizer
from transmux.polyphase_txmultiplexer import PolyphaseTxMultiplexer
from transmux.utils import gen_complex_chirp, gen_fdma, gen_complex_awgn


if __name__ == "__main__":
    num_channels = 7
    chanBW = 2000
    Fs = chanBW * num_channels

    data = gen_fdma(fs=Fs, bw=chanBW)
    # data = gen_complex_chirp(fs=Fs, duration=2)
    # data = gen_complex_awgn(size=Fs*10)

    num_blocks = 5
    block_len = int(np.floor(data.size / num_channels / num_blocks))
    inds = np.arange(0, block_len * num_channels * num_blocks, dtype=int).reshape(num_blocks, -1)

    # Create Trans-mux objects
    transition = 0.25
    rx = PolyphaseRxChannelizer(sample_rate_hz=Fs, channels=num_channels, transition=transition)
    print(rx)
    tx = PolyphaseTxMultiplexer(sample_rate_hz=Fs, channels=num_channels, transition=transition)
    print(tx)

    # Start the stopwatch / counter
    t1_start = perf_counter()
    for m in range(num_blocks):
        output = rx.process(data[inds[m, :]])
        if m == 0:
            nb_outputs = output
        else:
            nb_outputs = np.hstack((nb_outputs, output))
    # Stop the stopwatch / counter
    t1_stop = perf_counter()
    print("Elapsed time: {} s".format(t1_stop - t1_start))
    print("Samples per second: {}".format(np.prod(nb_outputs.shape) / (t1_stop - t1_start)))

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
    print("Samples per second: {}".format(wb_output.size / (t1_stop - t1_start)))


    # Channelizer plots
    plt.rcParams.update({'font.size': 7})
    # fig, ax = plt.subplots(num_channels + 3, 2)
    fig = plt.figure()
    gs = gridspec.GridSpec(num_channels+3, 2)

    t = np.arange(0, nb_outputs.shape[1]) / Fs * num_channels
    for n in range(0, rx.M):
        ax = fig.add_subplot(gs[n+1, 0])
        ax.specgram(nb_outputs[n, :], NFFT=128, Fs=Fs / num_channels, noverlap=100, vmin=-110)
        # ax.grid(True)
        ax.set_ylabel('C{}'.format(n))

        ax = fig.add_subplot(gs[n + 1, 1])
        ax.plot(t, 20 * np.log10(np.abs(nb_outputs[n, :])))
        ax.grid(True)
        ax.set_ylim(top=20)
        ax.set_ylim(bottom=-60)

    # Channelizer input plot (top)
    t = np.arange(0, data.size) / Fs
    ax = fig.add_subplot(gs[0, 0])
    ax.specgram(data, NFFT=256, Fs=Fs, noverlap=250, vmin=-110)
    # ax.grid(True)
    ax.set_ylabel('Original')
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, 20 * np.log10(np.abs(data)))
    ax.grid(True)
    ax.set_ylim(top=20)
    ax.set_ylim(bottom=-60)

    # Multiplexer output plot (bottom)
    t = np.arange(0, wb_output.size) / Fs
    ax = fig.add_subplot(gs[num_channels+1, 0])
    ax.specgram(wb_output, NFFT=256, Fs=Fs, noverlap=250, vmin=-110)
    # ax.grid(True)
    ax.set_ylabel('Recombined')
    ax = fig.add_subplot(gs[num_channels + 1, 1])
    ax.plot(t, 20 * np.log10(np.abs(wb_output)))
    ax.grid(True)
    ax.set_ylim(top=20)
    ax.set_ylim(bottom=-60)

    ax = fig.add_subplot(gs[num_channels + 2, :])
    f, Pxx_spec = signal.welch(data, Fs, 'flattop', 256, return_onesided=False, scaling='spectrum')
    ax.plot(np.fft.fftshift(f), 20*np.log10(np.abs(np.fft.fftshift(Pxx_spec))))
    f, Pxx_spec = signal.welch(wb_output[1000:], Fs, 'flattop', 256, return_onesided=False, scaling='spectrum')
    ax.plot(np.fft.fftshift(f), 20*np.log10(np.abs(np.fft.fftshift(Pxx_spec))))
    ax.set_xlabel('frequency [Hz]')
    # ax.set_ylabel('Linear spectrum [mW RMS]')
    ax.legend(('Original', 'Recombined'), loc='upper right')
    ax.grid(True)

    # plt.show()

    # Compare channel delay
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1)
    t_in = np.arange(0, data.size)
    t_out = np.arange(0, wb_output.size)
    delay = tx.filter.size - 0 + 0*rx.filter.size
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t_in, data.real, alpha=1.)
    ax.plot(t_out - delay, wb_output.real, alpha=1.)
    ax.legend(('Original', 'Recombined'), loc='upper right')
    ax.set_title('Filter bank time delay - real')
    ax.set_xlim(left=0)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_in, data.imag, alpha=1.)
    ax.plot(t_out - delay, wb_output.imag, alpha=1.)
    ax.legend(('Original', 'Recombined'), loc='upper right')
    ax.set_title('Filter bank time delay - imag')
    ax.set_xlim(left=0)
    plt.show()
