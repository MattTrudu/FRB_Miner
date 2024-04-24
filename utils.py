import sys
import os
import numpy as np


def cluster_channels(badchans):

    maxgap = 1
    badchans.sort()
    groups = [[badchans[0]]]
    for x in badchans[1:]:
        if abs(x - groups[-1][-1]) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups

def generate_dm_list(dm_start = 0, dm_end = 1000, dt = 1 * 10**(-6), ti = 1 * 10**(-3), f0 = 1000, df = 0.25, nchans = 128, tol = 1.25):

    """
    Code to generate Heimdall's DM list. Taken from [dedisp](https://github.com/ajameson/dedisp/blob/master/src/kernels.cuh#L56)
    Args:
        dm_start (float): Start DM
        dm_end (float): End DM
        dt (float): Sampling interval (in seconds)
        ti (float): pulse width (in seconds)
        f0 (float): Frequency of first channel (MHz)
        df (float): Channel Bandwidth (MHz)
        nchans (int): Number of channels
        tol (float): Tolerance level
    Returns:
        list : List of DMs for which Heimdall will do the search
    """
    dt *= 1e6
    ti *= 1e6
    center_freq = (f0 + (nchans / 2) * df) * 1e-3
    a = 8.3 * df / (center_freq ** 3)
    b = a ** 2 * nchans ** 2 / 16
    c = (dt ** 2 + ti ** 2) * (tol ** 2 - 1)

    dm_list = []
    dm_list.append(dm_start)
    while dm_list[-1] < dm_end:
        k = c + tol ** 2 * a ** 2 * dm_list[-1] ** 2
        dm = (
            b * dm_list[-1]
            + np.sqrt(-(a ** 2) * b * dm_list[-1] ** 2 + (a ** 2 + b) * k)
        ) / (a ** 2 + b)
        dm_list.append(dm)

    return dm_list

def dispersion_delay(fstart, fstop, DM = 0):

    """
    Simply computes the delay (in seconds) due to dispersion.The output will be, respectively, a scalar or a list of delays.
    """


    delay = 4148808.0 * DM * (1 / fstart ** 2 - 1 / fstop ** 2) / 1000

    return delay

def dedisperse_array(array, DM, freq, dt, ref_freq = "top"):
    """
    Dedisperse a dynamic spectrum array accoring to a DM and a reference frequency (top, central, bottom) of the bandwidth
    """

    k_DM = 1. / 2.41e-4
    dedisp_data = np.zeros_like(array)

    # pick reference frequency for dedispersion
    if ref_freq == "top":
        reference_frequency = freq[0]
    elif ref_freq == "center":
        center_idx = len(freq) // 2
        reference_frequency = freq[center_idx]
    elif ref_freq == "bottom":
        reference_frequency = freq[-1]
    else:
        print("`ref_freq` not recognized, using 'top'")
        reference_frequency = freq[0]

    shift = (k_DM * DM * (reference_frequency**-2 - freq**-2) / dt).round().astype(int)
    for i,ts in enumerate(array):
        dedisp_data[i] = np.roll(ts, shift[i])
    return dedisp_data
