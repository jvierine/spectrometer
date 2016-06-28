#!/usr/bin/env python

#
# Python interface to the spectrometer
#
# Juha Vierinen (2016)
#
import sys
import numpy as np
import ctypes
import ctypes as C
import matplotlib.pyplot as plt 

# find and load the library
lpl = ctypes.cdll.LoadLibrary("./libspectrometer.so")
# set the argument types for input to the wrapped function
lpl.process_echoes.argtypes = [C.POINTER(C.c_float), C.POINTER(C.c_float),\
                               C.c_int, C.c_int, C.c_int,\
                               C.POINTER(C.c_float),\
                               C.c_int, C.c_int, C.c_int]
# set the return type
lpl.process_echoes.restype = None

def process_echoes(tx_conj, echo, tx_length, ipp_length, n_ipp,spectrum, n_range_gates,\
                   range_gate_step, range_gate_start):
    """ Wrapper for process_echoes in plasmaline.cu and libplasmaline.so """

    lpl.process_echoes(tx.ctypes.data_as(C.POINTER(C.c_float)),\
                       echo.ctypes.data_as(C.POINTER(C.c_float)),\
                       C.c_int(tx_length),C.c_int(ipp_length),C.c_int(n_ipp),\
                       spectrum.ctypes.data_as(C.POINTER(C.c_float)),\
                       C.c_int(n_range_gates),C.c_int(range_gate_step),C.c_int(range_gate_start))

sr = 25 # range gate step size
def get_simulated_ipp(L=sr*10000,alt=sr*5000,freq=5e6,txlen_us=1000):
    """Simulated send and return signals. """

    txlen = txlen_us * sr
    echo = np.zeros(L,dtype=np.complex64)\
           + np.array((np.random.randn(L) + np.random.randn(L)*1j),\
                      dtype=np.complex64)

    tx = np.zeros(txlen, dtype=np.complex64)
    tx_bits = np.sign(np.random.randn(txlen_us))
    idx = np.arange(txlen_us) * sr

    for i in range(sr):
        tx[idx+i] = tx_bits

    tvec = np.arange(txlen) / (float(sr) * 1e6)

    upshifted = np.zeros(len(tvec), dtype=np.complex64)
    for i in range(50):
        upshifted += np.exp(1j * 1000.0 * np.random.randn(1))[0]\
                     * np.exp(1j * 2.0 * np.pi * (freq + np.random.randn(1)[0] * 1e3) * tvec)

    downshifted = np.zeros(len(tvec), dtype=np.complex64)
    for i in range(50):
        downshifted += np.exp(1j * 1000.0 * np.random.randn(1))[0]\
                       * np.exp(-1j * 2.0 * np.pi * (freq + np.random.randn(1)[0] * 1e3) * tvec)

    echo[alt:(alt+txlen)] = echo[alt:(alt+txlen)] + tx * (upshifted+downshifted)

    return((echo,tx))
 

if __name__ == "__main__":

    print "\nSimulated spectral analysis launching."

    # example parameters
    n_ipp = 100
    tx_length = 16384
    ipp_length = 250000
    n_range_gates = 4096
    range_gate_step = 25
    range_gate_start = 0
    tx = np.zeros([tx_length * n_ipp], dtype=np.complex64)
    echo = np.zeros([ipp_length * n_ipp], dtype=np.complex64)
    spectrum = np.zeros([tx_length * n_range_gates], dtype=np.float32)

    # simulate n_ipp plasma line echoes and transmit pulses
    print "Simulating data with CPU"
    for i in range(n_ipp):
        sys.stdout.write('.')
        sys.stdout.flush()
        (echo_ipp, tx_ipp) = get_simulated_ipp(L=25*10000, txlen_us=400, alt=0)
        tx[i * tx_length + np.arange(25*400)] = np.conj(tx_ipp)
        echo[i * ipp_length + np.arange(ipp_length)] = echo_ipp
    print "\nSimulation complete."

    process_echoes(tx, echo, tx_length, ipp_length, n_ipp, spectrum, n_range_gates,\
                   range_gate_step, range_gate_start)

    # check the simulation works as intended
    print "Checking for accuracy:"
    spectrum.shape = (n_range_gates, tx_length)
    shifted = np.fft.fftshift(spectrum[0,:])

    maxVal1 = max(shifted)
    maxIndex1 = np.argmax(shifted)
    shifted = np.delete(shifted, np.arange(maxIndex1 - 8, maxIndex1 + 8, 1))

    maxVal2 = max(shifted)
    maxIndex2 = np.argmax(shifted)
    shifted = np.delete(shifted, np.arange(maxIndex2 - 8, maxIndex2 + 8, 1))

    count = 0
    if (1e10 < maxVal1 < 1e12) and (1e10 < maxVal2 < 1e12):
        print "Passed spectrum maxima values test."
        count += 1
    else:
        print "Failed spectrum maxima values test."

    if all(i < 1e10 for i in shifted) is True:
        print "Passed spectrum elsewhere values test."
        count += 1
    else:
        print "Failed spectrum elsewhere values test."

    if (4900 < maxIndex1 < 4930) and (11440 < maxIndex2 < 11485):
        print "Passed spectrum maxima locations test."
        count += 1
    elif (4900 < maxIndex2 < 4930) and (11440 < maxIndex1 < 11485):
        print "Passed spectrum maxima locations test."
        count += 1
    else:
        print "Failed spectrum maxima locations test."

    if count == 3:
        print "\nPassed all tests. Working as intended.\n" 
    else:
        print "\nFailed at least one test. Troubleshooting needed.\n"


    ## plotting
    ### plasma line plot
    #spectrum.shape = (n_range_gates, tx_length)
    #plt.plot(np.linspace(-12.5, 12.5, num=tx_length), np.fft.fftshift(spectrum[0,:]))
    #plt.show()
    ### log color plot
    #smin = np.median(10.0 * np.log10(spectrum))
    #smax = np.max(10.0 * np.log10(spectrum))
    #plt.imshow(10.0 * np.log10(spectrum[2000:0:-1,:]), aspect = "auto",\
               #extent = [-12.5,12.5,0,1000], vmin = smin, vmax = 100,\
               #cmap = "nipy_spectral")
    ##plt.pcolormesh(spectrum)
    #plt.colorbar()
    #plt.show()

