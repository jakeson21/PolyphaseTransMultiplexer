import numpy as np
from scipy import signal
import sys


def make_polyphase_filter(channels, order=150, cutoff=0.5, transition_width=0.499):
    fs = channels
    # cutoff = 0.5
    # transition_width = 0.499
    num_taps, beta = signal.kaiserord(order, width=transition_width / (0.5 * fs))

    h = signal.firwin(num_taps, cutoff=cutoff, window=('kaiser', beta), scale=False, nyq=0.5 * fs)
    h[abs(h) <= 1e-15] = 0.
    h = h / np.max(abs(h))

    # form Hk
    size: int = int(np.ceil(np.ceil(h.size / fs) * fs))
    h.resize((size,), refcheck=False)
    # Columns are filter paths
    hp = np.reshape(h, (-1, fs)).transpose()
    return hp


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


def gen_complex_chirp(fs=44100, duration=1.):
    f0 = -fs / 2.1
    f1 = fs / 2.1
    t1 = duration
    beta = (f1 - f0) / float(t1)
    t = np.arange(0, t1, t1 / float(fs))
    data = np.exp(2j * np.pi * (.5 * beta * (t ** 2) + f0 * t))
    # Add noise to FDMA signal
    data += 0.001 * np.squeeze(np.random.randn(data.size, 2).view(np.complex128))
    return data


def gen_fdma(fs, bw):
    num_chans = int(np.floor(fs / bw))
    # Generate QPSK data of about 1 second
    C = np.asarray([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])
    S = C[np.random.randint(C.size, size=(bw*2, 1))]
    Sup = np.zeros((S.size, num_chans - 1,))
    x = np.hstack((S, Sup))
    data = np.squeeze(np.reshape(x, (x.size, -1)))
    hrrc = np.asarray(rcosdesign(beta=0.3, span=256, sps=num_chans * 2, name='normal'), dtype=np.cdouble)
    data = signal.lfilter(hrrc, 1, data)

    # generate fdma mapping
    f = np.linspace(-fs / 2, fs / 2, num_chans * 2 + 1)[1::2]
    fdma = np.random.randint(num_chans, size=10 * num_chans)
    fdma = np.sort(fdma)
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

