# TransmitChannelizer
TransmitChannelizer


## Volk test
Run time statistics:
```
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              12
On-line CPU(s) list: 0-11
Thread(s) per core:  2
Core(s) per socket:  6
Socket(s):           1
NUMA node(s):        1
Vendor ID:           AuthenticAMD
CPU family:          23
Model:               1
Model name:          AMD Ryzen 5 1600 Six-Core Processor
Stepping:            1
CPU MHz:             3172.300
BogoMIPS:            6387.13
Virtualization:      AMD-V
L1d cache:           32K
L1i cache:           64K
L2 cache:            512K
L3 cache:            8192K
NUMA node0 CPU(s):   0-11

16 GB DDR-4
```

Setup
- Running volk kernels for length 62500000 vectors
- Running FFTW with
  - FS = 62500000
  - blocks = 100
  - N = FS/blocks
  - NFFT = 2^nextpow2(N)

Output
```
Alignment is: 32
Running volk_32fc_s32fc_x2_rotator_32fc for length 62500000 vectors
Time elapsed in usec:  164669
Running volk_32fc_x2_add_32fc for length 62500000 vectors
Time elapsed in usec:  138388
Running fftw_execute 100 times for NFFT of 1048576
Time elapsed in usec:  638184
```
