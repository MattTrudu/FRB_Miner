#!/usr/bin/env python
import os
import sys
from sigpyproc.base import Filterbank
from sigpyproc.readers import FilReader
import time
import numpy as np
from scipy.signal import correlate
from scipy.linalg import toeplitz
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import argparse
import your
import warnings

# Ignore all warnings


def find_bad_bins(array, badchans_mask = None):

    """
    It takes a spectro-temporal data array and and renormalize the data with respect to a smoothed baseline and
    make some bin flagging with an IQR code.

    """

    renorm_data = np.copy(array)
    spec = renorm_data.mean(1)
    if badchans_mask is None:
        badchans_mask = np.zeros(spec.size, dtype=bool)

    for i, newd in enumerate(renorm_data):
        renorm_data[i, :] = (newd - spec[i]) / spec[i]

    baseline = np.mean(renorm_data[~badchans_mask, :], axis=0)
    badbins = np.zeros(baseline.size, dtype=bool)

    if baseline.size > int(0.1 * baseline.shape[0]):
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

    return badbins


def read_and_clean(filename,
                   output_dir = os.getcwd(),
                   output_name = None,
                   sk_sig = 5,
                   sg_sig = 5,
                   sg_win = 1,
                   clean_window = 0.1,
                   mode = "whitenose",
                   ):



    filedir, name = os.path.split(filename)

    if output_name is None:
        output_name = name.replace(".fil","") + "_cleaned" + ".fil"


    filterbank = FilReader(filename)

    nsamp = filterbank.header.nsamples
    nchan = filterbank.header.nchans
    nbits = filterbank.header.nbits
    df    = filterbank.header.foff
    dt    = filterbank.header.tsamp


    outfile = filterbank.header.prep_outfile(os.path.join(output_dir,output_name), back_compatible = True, nbits = nbits)
    channels = np.arange(0, nchan)

    sk_window = int(clean_window / dt)
    warnings.filterwarnings("ignore")
    for nsamps, ii, data in filterbank.read_plan(sk_window):
        #for out_file in enumerate(out_files):
            data = data.reshape(nsamps, filterbank.header.nchans)
            bad_chans = your.utils.rfi.sk_sg_filter(data, your.Your(filename), sk_sig, sg_win, sg_sig)
            bad_bins  = find_bad_bins(data.T, badchans_mask = bad_chans)
            mask = bad_bins[:, np.newaxis] | bad_chans
            if mode == "whitenoise":
                mu  = np.mean(data[~mask])
                std = np.std(data[~mask])
                data[mask] = np.abs(np.random.normal(mu,std))
            if mode == "zero":
                data[mask] = 0
            if int(nbits) == int(8):
                data = data.astype("uint8")
            if int(nbits) == int(16):
                data = data.astype("uint16")
            if int(nbits) == int(32):
                data = data.astype("uint32")
            outfile.cwrite(data.ravel())


    outfile.close()

    return outfile.name

def _get_parser():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description = "Clean a SIGPROC filterbank file from RFI and produces a cleaned filterbank" + "\n"
                      "It performs an RFI excision in frequency via spectral kurtosis " + "\n"
                      "It performs an RFI excision in time via an IQR " + "\n"
                      "It works only with > 8-bits filterbanks...")
    parser.add_argument('-f', '--file', help="SIGPROC .fil file (required)", required = True)


    parser.add_argument(
        "-o",
        "--output_dir",
        help = "Output directory (Default: current directory)",
        type = str,
        default = os.getcwd(),
    )


    parser.add_argument('-n',
                        '--output_name',
                        action = "store" ,
                        help = "Output File Name (Default: filename_cleaned.fil)",
                        default = None
                        )
    parser.add_argument('-m',
                        '--mode',
                        type = str,
                        action = "store" ,
                        help = "Mode to substitute the data (whitenoise or zero)",
                        default = "whitenoise"
                        )
    parser.add_argument('-cl_win',
                        '--clean_window',
                        type = float,
                        default = 0.1,
                        action = "store" ,
                        help = "Window (in s) of data to read the data and evaluate the statistics."
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
    return parser.parse_args()

if __name__ == '__main__':

    args = _get_parser()

    read_and_clean(args.file,
                    output_dir = args.output_dir,
                    output_name = args.output_name,
                    sk_sig = args.sk_sigma,
                    sg_sig = args.sg_sigma,
                    sg_win = args.sg_window,
                    clean_window = args.clean_window,
                    mode = args.mode,
                    )
