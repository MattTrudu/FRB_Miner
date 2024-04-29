#!/usr/bin/env python
import numpy as np
import your
import os
import itertools
import argparse
from utils import plan_subbands



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
    parser.add_argument(
        "-s",
        "--save",
        help = "Save subbands",
        default = False,
    )
    return parser.parse_args()



if __name__ == '__main__':

    args = _get_parser()

    plan_subbands(args.file,
                  fthresh = args.band_threshold,
                  overlap = args.overlap,
                  output_dir = args.output_dir,
                  save = args.save,
                  output_name = args.name)
