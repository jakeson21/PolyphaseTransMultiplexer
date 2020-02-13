#!/usr/bin/env python3
def make_polyphase_filter(channels, order=150):
    fs = channels
    cutoff = 0.5
    transition_width = 0.499
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
