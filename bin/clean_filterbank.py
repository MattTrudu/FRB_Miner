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
import matplotlib
import matplotlib.pyplot as plt
cmap = matplotlib.colors.ListedColormap(['white', 'black'])
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

def scale_array_to_range(array, nbits = 8):
    """
    Scale a processed 2D array to the range of 0 to 2^nbits-1.

    Parameters:
    - array: NumPy array, the processed 2D array to be scaled.
    - nbits: int, the number of bits of the data.

    Returns:
    - scaled_array: NumPy array, the scaled array within the specified range.
    """
    # Renormalise data

    # Compute the maximum value that can be represented with nbits
    max_value = 2**nbits - 1

    # Scale the data to the range [-1, 1]
    min_val = np.min(array)
    max_val = np.max(array)
    scaled_array = (2 * (array - min_val) / (max_val - min_val)) - 1

    # Map the range [-1, 1] to [0, max_value]
    scaled_array = ((scaled_array + 1) / 2) * max_value

    # Convert the scaled array to integer type with nbits
    return scaled_array

def eigenbasis(matrix):

    """
    Compute the eigenvalues and the eigenvectors of a square matrix and return the eigenspectrum (eigenvalues sorted in decreasing order) and the sorted eigenvectors respect
    to the eigenspectrum for the KLT analysis
    """

    eigenvalues,eigenvectors = np.linalg.eigh(matrix)

    if eigenvalues[0] < eigenvalues[-1]:
        eigenvalues = np.flipud(eigenvalues)
        eigenvectors = np.fliplr(eigenvectors)
    eigenspectrum = eigenvalues
    return eigenspectrum,eigenvectors

def count_elements_for_threshold(arr, threshold):
    sorted_arr = np.sort(arr)[::-1]  # Sort array in descending order
    total_sum = np.sum(sorted_arr)
    cumulative_sum = np.cumsum(sorted_arr)
    num_elements = np.searchsorted(cumulative_sum, threshold * total_sum, side='right') + 1
    return num_elements

def klt(signals, threshold):

    R = np.cov((signals-np.mean(signals,axis=0)),rowvar=False)

    eigenspectrum,eigenvectors = eigenbasis(R)

    neig = count_elements_for_threshold(eigenspectrum, threshold)

    coeff = np.matmul((signals[:,:]-np.mean(signals,axis=0)),np.conjugate((eigenvectors[:,:])))
    recsignals = np.matmul(coeff[:,0:int(neig)],np.transpose(eigenvectors[:,0:int(neig)])) + np.mean(signals,axis=0)

    return eigenspectrum,eigenvectors,recsignals

def read_and_clean(filename,
                   output_dir = os.getcwd(),
                   output_name = None,
                   sk_clean = False,
                   sk_sig = 5,
                   sg_sig = 5,
                   sg_win = 1,
                   klt_clean = False,
                   klt_thr = 0.4,
                   z_thr = 1,
                   clean_window = 0.1,
                   mode = "whitenoise",
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
        data = data.reshape(nsamps, filterbank.header.nchans)
        if ii in [0,1,2,3,4]:
            print(data.mean(),data.std())
        if sk_clean:
            bad_chans = your.utils.rfi.sk_sg_filter(data, your.Your(filename), sk_sig, sg_win, sg_sig)
            if mode == "whitenoise":
                data[:,bad_chans] = np.random.normal(data[:,~bad_chans].mean(),data[:,~bad_chans].std(), size = bad_chans.sum())
                data = scale_array_to_range(data, nbits = nbits)
            elif mode == "zero":
                data[:,bad_chans] = 0
            else:
                ValueError("Mode can be either whitenoise or zero")
        if klt_clean:
            eigenspectrum,eigenvectors,kltdata = klt(data, klt_thr)
            if ii in [0,1,2,3,4]:
                print(kltdata.mean(),kltdata.std())
            z_scores = (kltdata - np.mean(kltdata)) / np.std(kltdata)
            outliers_mask = np.abs(z_scores) > z_thr
            if mode == "whitenoise":
                mu  = data[~outliers_mask].mean()
                std = data[~outliers_mask].std()
                data[outliers_mask] = np.random.normal(mu, std, size = outliers_mask.sum())
                data = scale_array_to_range(data, nbits = nbits)
            elif mode == "zero":
                data[outliers_mask] = 0
            else:
                ValueError("Mode can be either whitenoise or zero")

        else:
            ValueError("Cleaning strategy not picked...")

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
                        help = "Mode to substitute the data in the flagging (whitenoise or zero)",
                        default = "zero"
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
    parser.add_argument(
                        "-sk",
                        "--sk_clean",
                        help = "Find the bad channels with a instantaneous spectral kurtosis",
                        action = "store_true",
                        )
    parser.add_argument(
                        "-klt",
                        "--klt_clean",
                        help = "Create an RFI template via the KLT and subtract it from the data",
                        action = "store_true",
                        )
    parser.add_argument('-klt_th',
                        '--klt_threshold',
                        type = float,
                        default = 0.4,
                        action = "store" ,
                        help = "Percentage, from 0 (doing nothing) to 1 (removing all the data) of the total variance of signal to evaluate the number of eigen images for the KLT RFI template"
                        )
    parser.add_argument('-z_th',
                        '--z_threshold',
                        type = float,
                        default = 1.0,
                        action = "store" ,
                        help = "Z-score threshold to get the bad pixels from the KLT RFI template"
                        )
    return parser.parse_args()

if __name__ == '__main__':

    args = _get_parser()

    read_and_clean(args.file,
                    output_dir = args.output_dir,
                    output_name = args.output_name,
                    sk_clean = args.sk_clean,
                    sk_sig = args.sk_sigma,
                    sg_sig = args.sg_sigma,
                    sg_win = args.sg_window,
                    clean_window = args.clean_window,
                    mode = args.mode,
                    klt_clean = args.klt_clean,
                    klt_thr = args.klt_threshold,
                    z_thr = args.z_threshold)
