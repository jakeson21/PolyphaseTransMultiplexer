import numpy as np
from scipy import signal


def pareq(G, GB, w0, B):
    """
    [num, den] = pareq(G, GB, w0, B)
    Second-order parametric equalizing filter design
    with adjustable bandwidth gain

    [num, den] = pareq(G, GB, w0, B)

    Parameters
    G = peak gain (linear)
    GB = bandwidth gain (linear)
    w0 = center frequency (rads/sample)
    B = bandwidth (rads/sample)

    Output
    num = [b0, b1, b2] = numerator coefficients
    den = [1,  a1, a2] = denominator coefficients

    Written by Vesa Valimaki, August 24, 2016
    Ref. S. Orfanidis, Introduction to Signal Processing, p. 594
    We have set the dc gain to G0 = 1.
    """

    if G == 1:
        beta = np.tan(B / 2)  # To avoid division by zero, when G=1
    else:
        beta = np.sqrt(np.abs(GB ** 2. - 1.) / np.abs(G ** 2. - GB ** 2.)) * np.tan(B / 2.)

    num = np.asarray(((1. + G * beta), -2 * np.cos(w0), (1. - G * beta))) / (1. + beta)
    den = np.asarray((1., -2. * np.cos(w0) / (1. + beta), (1. - beta) / (1. + beta)))

    return num, den


def db(x):
    return 10. * np.log10(x)


def interactionMatrix(G, gw, wg, wc, bw):
    """
    Compute the interaction matrix of a cascade graphic equalizer containing
    the leak factors to account for the band interaction when assigning
    filter gains. All filters are Orfanidis peak/notch filters with
    adjustable bandwidth gain.

    Input parameters:
    :param G  = Linear gain at which the leakage is determined
    :param gw = Gain factor at bandwidth (0.5 refers to db(G)/2)
    :param wg = Command frequencies i.e. filter center frequencies (in rad/sample)
    :param wc = Design frequencies (rad/sample) at which leakage is computed
    :param bw = Bandwidth of filters in radians

    Output:
    :return: leak = N by M matrix showing how the magnitude responses of the
    band filters leak to the design frequencies. N is determined from the
    length of the array wc (number of design frequencies) whereas M is
    determined from the length of wg (number of filters)

    Uses pareq.m

    Written by Vesa Valimaki, Espoo, Finland, 12 April 2016
    Modified by Juho Liski, Espoo, Finland, 26 September 2016

    Aalto University, Dept. of Signal Processing and Acoustics
    """

    M = wg.size  # The number of center frequencies (filters)
    N = wc.size  # The number of design frequencies
    leak = np.zeros((M, N))  # Initialize interaction matrix
    Gdb = db(G)  # Convert linear gain factor to dB
    Gdbw = gw * Gdb  # dB gain at bandwidth
    Gw = 10. ** (Gdbw / 20.)  # Convert to linear gain factors

    # Estimate leak factors of peak/notch filters
    for m in range(0, M):  # Filters
        [num, den] = pareq(G[m], Gw[m], wg[m], bw[m])  # Parametric EQ filter
        f, H = signal.freqz(num, den, wc)  # Evaluate response at wc frequencies
        Gain = db(H) / Gdb[m]  # Normalized interference (Re 1 dB)
        leak[m, :] = np.abs(Gain)  # Store gain in a matrix
    return leak


def acge3(Gdb):
    """
    Design third-octave EQ according to the method presented by J. Liski and
    V. Valimaki in "The Quest for the Best Graphic Equalizer," in Proc.
    DAFx-17, Edinburgh, UK, Sep. 2017.

    Input parameters:
    :param Gdb  = command gains in dB, size 31x1

    Output:
    :return: numsopt = numerator parts of the 31 filters
    :return: densopt = denominator parts of the 31 filters

    Uses pareq.m and interactionMatrix.m

    Created by Juho Liski and Vesa Valimaki, Otaniemi, Espoo, Finland, 21 June 2017
    Modified by Juho Liski, Otaniemi, Espoo, Finland, 6 May 2019

    Aalto University, Dept. of Signal Processing and Acoustics
    Gdb:
    """
    fs = 44.1e3  # Sample rate
    fc1 = np.asarray([19.69, 24.80, 31.25, 39.37, 49.61, 62.50, 78.75, 99.21, 125.0, 157.5, 198.4,
                      250.0, 315.0, 396.9, 500.0, 630.0, 793.7, 1000, 1260, 1587, 2000, 2520, 3175, 4000,
                      5040, 6350, 8000, 10080, 12700, 16000, 20160])  # Log center frequencies for filters
    fc2 = np.zeros((61,))  # Center frequencies and intermediate points between them
    fc2[0:61:2] = fc1
    for k in range(1, 61, 2):
        fc2[k] = np.sqrt(fc2[k - 1] * fc2[k + 1])  # Extra points are at geometric mean frequencies

    wg = 2 * np.pi * fc1 / fs  # Command gain frequencies in radians
    wc = 2 * np.pi * fc2 / fs  # Center frequencies in radians for iterative design with extra points
    gw = 0.4  # Gain factor at bandwidth (parameter c)
    bw = 2 * np.pi / fs * np.asarray([9.178, 11.56, 14.57, 18.36, 23.13, 29.14, 36.71, 46.25, 58.28, 73.43,
                                      92.51, 116.6, 146.9, 185.0, 233.1, 293.7, 370.0, 466.2, 587.4, 740.1, 932.4,
                                      1175, 1480, 1865, 2350, 2846, 3502, 4253, 5038, 5689, 5570])  # EQ filter bandwidths

    leak = interactionMatrix(10. ** (17. / 20.) * np.ones((1, 31)), gw, wg, wc, bw)  # Estimate leakage b/w bands
    Gdb2 = np.zeros((61, 1))
    Gdb2[0:61:2] = Gdb
    for k in range(1, 61, 2):
        Gdb2[k] = (Gdb2[k - 1] + Gdb2[k + 1]) / 2.  # Interpolate target gains linearly b/w command gains

    Goptdb = np.linalg.lstsq(leak, Gdb2)  # ldivide - Solve first estimate of dB gains based on leakage
    Gopt = 10. ** (Goptdb / 20.)  # Convert to linear gain factors

    # Iterate once
    leak2 = interactionMatrix(Gopt, gw, wg, wc, bw)  # Use previous gains
    G2optdb = np.linalg.lstsq(leak2, Gdb2)  # ldivide - Solve optimal dB gains based on leakage
    G2opt = 10. ** (G2optdb / 20.)  # Convert to linear gain factors
    G2woptdb = gw * G2optdb  # Gain at bandwidth wg
    G2wopt = 10. ** (G2woptdb / 20.)  # Convert to linear gain factor

    # Design filters with optimized gains
    numsopt = np.zeros((3, 31))  # 3 num coefficients for each 10 filters
    densopt = np.zeros((3, 31))  # 3 den coefficients for each 10 filters
    for k in range(32):
        [num, den] = pareq(G2opt[k], G2wopt[k], wg[k], bw[k])  # Design filters
        numsopt[:, k] = num
        densopt[:, k] = den

    return numsopt, densopt
