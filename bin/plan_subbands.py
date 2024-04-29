#!/usr/bin/env python
import numpy as np
import your
import os
import itertools
import argparse

def plan_subbands(filename, fthresh = 100, overlap=False, output_dir = os.getcwd(),output_name = "subbands.npy"):

    filfile = your.Your(filename)

    tsamp     = filfile.your_header.native_tsamp
    nsamp     = filfile.your_header.native_nspectra
    nchan     = filfile.your_header.native_nchans
    foff      = filfile.your_header.foff
    fch0      = filfile.your_header.fch1
    tstartutc = filfile.your_header.tstart_utc

    bw = nchan * foff
    fc = fch0 + bw/2
    freqs = np.arange(fc - bw / 2, fc + bw / 2, foff)

    flo = freqs.min()
    fhi = freqs.max()

    channels = np.arange(nchan)

    subbands = []
    s = 1 if overlap else 0

    for level in itertools.count(s):
        f_delim = np.linspace(flo**(-2), fhi**(-2), 2**level+1)**(-0.5)


        for i in range(2**level-s):
            f0, f1 = f_delim[i], f_delim[i+s+1]
            if f1-f0 >= fthresh:
                subbands.append((nchan - 1 - np.searchsorted(freqs[::-1], f1), nchan - 1 - np.searchsorted(freqs[::-1], f0)))
                #subbands.append((f0, f1))
                #yield f0, f1



        subbands = np.array(subbands)
        print("Subbands:")
        print(subbands)

        np.save(os.path.join(output_dir, output_name),subbands)


def _get_parser():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser(
        prog="plan_subbads.py",
        description = "Compute the optimal sub-bands to be processed given a certain bandiwidth threshold of interest. It reads a filterbank and returns a npy array with the channels to be processed.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,

    )
    parser.add_argument(
        '-f',
        '--file',
        help = "SIGPROC .fil file (required)",
        required = True,
    )
    parser.add_argument(
        "-b",
        "--band_threshold",
        help = "Minimum bandwidth (in MHz) to be searched",
        type = float,
        default = 100,
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        help = "Output directory",
        type = str,
        default = os.getcwd(),
    )

    parser.add_argument(
        "-n",
        "--name",
        help = "Name of the npy file",
        type = str,
        default = "subbands.npy",
    )

    parser.add_argument(
        "-ov",
        "--overlap",
        help = "Create also overlapped subbands",
        default = False,
    )
    return parser.parse_args()



if __name__ == '__main__':

    args = _get_parser()

    plan_subbands(args.file,
                  fthresh = args.band_threshold,
                  overlap = args.overlap,
                  output_dir = args.output_dir,
                  output_name = args.name)
