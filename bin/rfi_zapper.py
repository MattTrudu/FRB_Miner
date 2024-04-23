#!/usr/bin/env python
import numpy as np
import os
import sys
import your
import argparse
import getpass
from datetime import datetime
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import detrend

def renormalize_data(array, badchans_mask = None):

    """
    It takes a spectro-temporal data array and and renormalize the data with respect to a smoothed baseline and make some flagging with an IQR code.
    """


    renorm_data = np.copy(array)
    spec = renorm_data.mean(1)
    if badchans_mask is None:
        badchans_mask = np.zeros(spec.size, dtype=bool)

    for i, newd in enumerate(renorm_data):
        renorm_data[i, :] = (newd - spec[i]) / spec[i]

    baseline = np.mean(renorm_data[~badchans_mask, :], axis=0)
    badbins = np.zeros(baseline.size, dtype=bool)

    if baseline.size > 1000:
        smooth_baseline = gaussian_filter(baseline, 101, truncate = 1)
        detr = baseline - smooth_baseline

        ordered = np.sort(detr)
        q1 = ordered[baseline.size // 4]
        q2 = ordered[baseline.size // 2]
        q3 = ordered[baseline.size // 4 * 3]
        lowlim = q2 - 2 * (q2 - q1)
        hilim = q2 + 2 * (q3 - q2)

        badbins = (detr < lowlim) | (detr > hilim)
        baseline = smooth_baseline

    renorm_data[~badchans_mask, :] -= baseline
    renorm_data[badchans_mask, :] = 0
    renorm_data[:, badbins] = 0

    return renorm_data

def iqr_filter(data, badchans_mask = None):

    """
    Flag noisier channels considering a positive-negative spectrum with a IQR algorithm and add a previous mask.
    """

    newdata = np.copy(data)

    if badchans_mask is None:
        badchans_mask = np.zeros(newdata.size, dtype=bool)

    spec = np.std(newdata, axis=1)

    ordered = np.sort(spec[~badchans_mask])

    q1 = ordered[ordered.size // 4]
    q2 = ordered[ordered.size // 2]
    q3 = ordered[ordered.size // 4 * 3]
    lowlim = q2 - 2 * (q2 - q1)
    hilim = q2 + 2 * (q3 - q2)

    badchans = (spec < lowlim) | (spec > hilim) | badchans_mask

    return badchans

def dedisperse(wfall, DM, freq, dt, ref_freq="top"):
    """
    Dedisperse a wfall matrix to DM.
    """

    k_DM = 1. / 2.41e-4
    dedisp = np.zeros_like(wfall)

    # pick reference frequency for dedispersion
    if ref_freq == "top":
        reference_frequency = freq[-1]
    elif ref_freq == "center":
        center_idx = len(freq) // 2
        reference_frequency = freq[center_idx]
    elif ref_freq == "bottom":
        reference_frequency = freq[0]
    else:
        #print "`ref_freq` not recognized, using 'top'"
        reference_frequency = freq[-1]

    shift = (k_DM * DM * (reference_frequency**-2 - freq**-2) / dt).round().astype(int)
    for i,ts in enumerate(wfall):
        dedisp[i] = np.roll(ts, shift[i])
    return dedisp

def rfi_mask(data, badchans):

    data = np.asarray(data , dtype = np.float32)

    data = renormalize_data(data, badchans_mask = badchans)

    badchans = iqr_filter(data, badchans_mask = badchans) | badchans

    return badchans

def flag_rfi(filename,
             outdir = os.getcwd(),
             name = "mask",
             tstart = 0,
             ngulp  = 8192,
             sk_sig = 5,
             sg_sig = 5,
             sg_win = 1,
             verbosity = None,
             plot = None):

    filfile = your.Your(filename)

    tsamp     = filfile.your_header.native_tsamp
    nsamp     = filfile.your_header.native_nspectra
    nchan     = filfile.your_header.native_nchans
    foff      = filfile.your_header.foff
    fch0      = filfile.your_header.fch1
    tstartutc = filfile.your_header.tstart_utc

    bw = nchan * foff
    fc = fch0 + bw/2
    obslen = nsamp*tsamp

    freqs = np.arange(fc - bw / 2, fc + bw / 2, foff)
    times = np.linspace(0,obslen,int(nsamp))
    channels = np.arange(nchan)


    nstart = np.rint(tstart / tsamp).astype(np.int)

    if verbosity:
        print(f"Opening data. Getting {ngulp} samples...")

    data = filfile.get_data(nstart = nstart, nsamp = ngulp)

    if verbosity:
        print(f"Finding bad channels...")

    mask = your.utils.rfi.sk_sg_filter(data, filfile, sk_sig, sg_win, sg_sig)

    mask = rfi_mask(data,mask)

    if verbosity:
        print(f"Done.")
        print(f"Masked the following channels: ")
        print(f"{channels[mask]}")
        print(f"Saving the mask...")

    maskname = os.path.join(outdir,name)

    maskname = maskname + f".bad_chans"

    with open(maskname, "w") as f:
        np.savetxt(f, channels[mask], fmt="%d", delimiter = " ", newline = " ")

    if plot:
        if verbosity:
            print("Making a plot of the bandpass...")

        data = np.asarray(data.T , dtype = np.float32)
        psd  = np.abs(data.mean(1))
        badchans = mask

        figure = plt.figure(figsize = (20,10))
        #plt.style.use('dark_background')
        mpl.rcParams['axes.linewidth'] = 1.0

        gs = plt.GridSpec(1,1,hspace = 0.05 , wspace = 0,  width_ratios = [1], height_ratios = [1], top = 0.85, bottom = 0.15, right = 0.95, left = 0.05)

        ax0 = plt.subplot(gs[0,0])
        ax1 = ax0.twiny()

        size = 20

        ax0.set_yticks([])
        ax0.set_xticks([freqs[0], freqs[0] + bw / 8, freqs[0] + bw / 4 , freqs[0] + 3 * bw / 8 , freqs[0] + bw / 2 ,freqs[0] + 5 * bw / 8, freqs[0] + 3 * bw / 4,freqs[0] + 7 * bw / 8 ,freqs[-1]])
        ax1.set_xticks([nchan, int(7 * nchan / 8) ,int(3 * nchan / 4) ,int(5 * nchan / 8), int(nchan / 2),int(3 * nchan / 8), int(nchan / 4),int(1 * nchan / 8), 0])
        ax1.set_xticklabels([nchan, int(7 * nchan / 8) ,int(3 * nchan / 4) ,int(5 * nchan / 8), int(nchan / 2),int(3 * nchan / 8), int(nchan / 4),int(1 * nchan / 8), 0])


        ax0.tick_params(which='major', labelsize  = size)
        ax1.tick_params(which='major', labelsize  = size)

        ax1.set_xlim(nchan,0)


        ax0.margins(x = 0)
        ax1.margins(x = 0)


        ax0.set_xlabel("Frequency (MHz)" , size = size)
        ax0.set_ylabel("Flux (Arbitrary Units)" , size = size)
        ax1.set_xlabel("Channels" , size = size)


        ax0.plot(freqs , psd, color = "black",  linewidth = 2, label = "Bandpass")
        ax0.plot(freqs[badchans] , psd[badchans] , "s" , color = "darkred" , linewidth = 2, label = "Flagged Channels")
        ax0.legend(loc = 0 , fontsize = size)

        username = getpass.getuser()
        datetimenow = datetime.utcnow()

        figure.text(0.01,0.95,"Plot made by %s on %s UTC"%(username,str(datetimenow)[0:19]),fontsize = size - 5)

        figname = os.path.join(outdir,name)

        figname = figname + "_rfi_mask.png"

        figure.savefig(figname)

def _get_parser():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser(
        prog="rfiflag.py",
        description = "Find noisest channels given a SIGPROC .fil file. It produce a txt file (.bad_chans) containg the flagged channels. An optional plot of the bandpass is available.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,

    )
    parser.add_argument('-f', '--file', help="SIGPROC .fil file (required)", required = True)

    parser.add_argument(
        "-o",
        "--output_dir",
        help = "Output directory (Default: current directory)",
        type = str,
        default = os.getcwd(),
    )

    parser.add_argument(
        "-n",
        "--mask_name",
        help = "Name of the mask file",
        type = str,
        default = "mask",
    )

    parser.add_argument(
        "-tstart",
        "--time_start",
        help = "Starting time (in s) to which grab the data",
        type = float,
        default = 0,
    )

    parser.add_argument(
        "-ngulp",
        "--nsamps_gulp",
        help = "Number of samples to be read",
        type = int,
        default = 8192,
    )

    parser.add_argument(
        "-sksig",
        "--sk_sigma",
        help = "Spectral Kurtosis sigma",
        type = int,
        default = 5,
    )

    parser.add_argument(
        "-sgsig",
        "--sg_sigma",
        help = "Savitzky-Golay sigma",
        type = int,
        default = 5,
    )

    parser.add_argument(
        "-sgwin",
        "--sg_window",
        help = "Savitzky-Golay window (in MHz)",
        type = int,
        default = 15,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        help = "Be verbose",
        action = "store_true",
    )

    parser.add_argument(
        "-p",
        "--plot",
        help = "Make a bandpass plot",
        action = "store_true",
    )

    return parser.parse_args()



if __name__ == '__main__':

    args = _get_parser()

    flag_rfi(args.file,
             outdir = args.output_dir,
             name = args.mask_name,
             tstart = args.time_start,
             ngulp  = args.nsamps_gulp,
             sk_sig = args.sk_sigma,
             sg_sig = args.sg_sigma,
             sg_win = args.sg_window,
             verbosity = args.verbose,
             plot = args.plot)
