#!/usr/bin/env python
import logging
import sys
import os
import textwrap
import argparse
import numpy as np
import your
import subprocess



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


def launch_heimdall(filename,
                    dm_tol=None,
                    ngulp=None,
                    boxcar_max=None,
                    baseline_length=None,
                    DM=None,
                    mask=None,
                    rfi_no_narrow=None,
                    rfi_no_broad=None,
                    no_scrunching=None,
                    rfi_tol=None,
                    gpu_id=None,
                    verbosity=None,
                    scrunch_tol=None,
                    outdir=None,
                    fswap=False):

    filfile = your.Your(filename)

    tsamp     = filfile.your_header.native_tsamp
    nsamp     = filfile.your_header.native_nspectra
    nchan     = filfile.your_header.native_nchans
    foff      = filfile.your_header.foff
    fch0      = filfile.your_header.fch1
    tstartutc = filfile.your_header.tstart_utc

    bw = nchan * foff
    fc = fch0 + bw / 2
    obslen = nsamp * tsamp

    cmd = "heimdall"
    cmd += f" -f {filename}"

    if DM:
        if DM[1] < DM[0]:
            logging.warning("Second DM value is less than the first: swapping...")
        cmd += f" -dm {min(DM)} {max(DM)}"

    if baseline_length:
        cmd += f" -baseline_length {baseline_length}"

    if boxcar_max:
        bmax = int(np.rint((boxcar_max * 1e-3) / tsamp))  # boxcar_max in ms -> bins
        cmd += f" -boxcar_max {bmax}"

    if ngulp:
        cmd += f" -nsamps_gulp {ngulp}"

    if fswap:
        cmd += " -fswap"

    if rfi_tol:
        cmd += f" -rfi_tol {rfi_tol}"

    if rfi_no_narrow:
        cmd += " -rfi_no_narrow"

    if rfi_no_broad:
        cmd += " -rfi_no_broad"

    if no_scrunching:
        cmd += " -no_scrunching"

    if dm_tol:
        cmd += f" -dm_tol {dm_tol}"

    if gpu_id:
        cmd += f" -gpu_id {gpu_id}"

    if verbosity:
        cmd += f" -{verbosity}"

    if outdir:
        cmd += f" -output_dir {outdir}"

    if scrunch_tol:
        cmd += f" -scrunch_tol {scrunch_tol}"  # FIX: previously was -output_dir again!

    if mask:
        try:
            mask_data = np.loadtxt(mask, dtype=np.int32)

            if mask_data.size == 0:
                pass  # empty mask; do nothing
            elif len(mask_data.shape) == 1:
                badchans = mask_data
                groups = cluster_channels(badchans)  # assume this groups adjacent channels
                for group in groups:
                    cmd += f" -zap_chans {np.min(group)} {np.max(group)}"
            else:
                logging.warning("RFI mask not understood; expected 1D. Ignoring.")
        except Exception as e:
            logging.warning(f"Failed to parse RFI mask file: {e}")

    return cmd



def _get_parser():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser(
        prog="launch_heimdall.py",
        description = "Python wrapper for Heimdall. It allows to use a txt file (.bad_chans) in which are present the channels to flag the RFI.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,

    )
    parser.add_argument(
        '-f',
        '--file',
        help = "SIGPROC .fil file (required)",
        required = True,
    )

    parser.add_argument(
        "-dm",
        "--dm",
        help = "Dispersion Measure (pc cm^-3) (Default: 0 1000)",
        type = float,
        nargs = 2,
        default = None,
    )

    parser.add_argument(
        "-dm_tol",
        "--dm_tolerance",
        help = "SNR loss tolerance between DM trials ",
        type = float,
        default = 1.25,
    )

    parser.add_argument(
        "-box_max",
        "--boxcar_max",
        help = "Maximum boxcar width (in ms) to use for the matched filtering (Default: 10 ms)",
        type = float,
        default = None,
    )

    parser.add_argument(
        "-base_len",
        "--baseline_length",
        help = "Number of seconds over which to smooth the baseline (Default: 2 s)",
        type = float,
        default = None,
    )

    parser.add_argument(
        "-gpu_id",
        "--gpu_id",
        help = "Specify the gpu number to which run Heimdall",
        type = int,
        default = None,
    )


    parser.add_argument(
        "-rfi_no_narrow",
        "--rfi_no_narrow",
        help="Disable narrow band RFI excision (Default: don't use this option)",
        required=False,
        action="store_true",
        default = None,
    )

    parser.add_argument(
        "-rfi_no_broad",
        "--rfi_no_broad",
        help="Disable 0-DM RFI excision (Default: don't use this option)",
        required=False,
        action="store_true",
        default = None,
    )

    parser.add_argument(
        "-no_scrunching",
        "--no_scrunching",
        help = "Don't use an adaptive time scrunching during dedispersion (Default: don't use this option)",
        required = False,
        action="store_true",
        default = None,
    )

    parser.add_argument(
        "-scrunch_tol",
        "--scrunching_tol",
        help = "Smear tolerance factor for time scrunching (Default: 1.5) ",
        type = float,
        default = None,
    )

    parser.add_argument(
        "-rfi_tol",
        "--rfi_tol",
        help = "RFI exicision threshold limits (Default: 5)",
        type = int,
        default = None,
    )

    parser.add_argument(
        "-ngulp",
        "--nsamps_gulp",
        help = "Number of samples to be read at a time (Default: 262144)",
        type = int,
        default = None,
    )

    parser.add_argument(
        "-fswap",
        "--fswap",
        help = "Swap channel ordering for negative DM - SIGPROC 2,4 or 8 bit only",
        type = bool,
        default = False,
    )

    parser.add_argument(
        "-verb",
        "--verbosity",
        help = "Heimdall verbosity (Options: -v -V -g -G, Default: -v)",
        type = str,
        default = None,
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        help = "Heimdall output directory (Default: current directory)",
        type = str,
        default = None,
    )

    parser.add_argument(
        "-m",
        "--mask_file",
        help = "Mask file (.bad_chans) to zap channels (Default: no mask)",
        type = str,
        default = None,
    )

    return parser.parse_args()

if __name__ == '__main__':

    args = _get_parser()

    cmd = launch_heimdall(args.file, dm_tol = args.dm_tolerance,
                        ngulp = args.nsamps_gulp,
                        boxcar_max = args.boxcar_max,
                        baseline_length = args.baseline_length,
                        DM = args.dm,
                        mask = args.mask_file,
                        rfi_no_narrow = args.rfi_no_narrow,
                        rfi_no_broad = args.rfi_no_broad,
                        no_scrunching = args.no_scrunching,
                        rfi_tol = args.rfi_tol,
                        gpu_id = args.gpu_id,
                        verbosity = args.verbosity,
                        scrunch_tol = args.scrunching_tol,
                        outdir = args.output_dir,
                        fswap = args.fswap)


    os.system(cmd)
