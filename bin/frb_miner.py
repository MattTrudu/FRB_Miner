#!/usr/bin/env python
import os
import sys 
import numpy as np
import yaml
import argparse

def main(args):

    filename   = args.file
    configfile = args.config
    outdir     = args.output_dir
    config_data = yaml.safe_load(configfile)

    subband_search = config_data['subband_search']

    print(subband_search)

def _get_parser():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser(
        prog = "frb_miner.py",
        description = "Python FRB searcher. It performs RFI excision, candidate search and classification. The options should be parsed via a config YAML file.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,

    )
    parser.add_argument(
        '-f',
        '--file',
        type = str,
        help = "SIGPROC .fil file (required)",
        required = True,
    )
    parser.add_argument(
        '-c',
        '--config',
        type = str,
        help = "YAML config file for the search",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help = "Output directory",
        type = str,
        default = os.getcwd(),
    )
    return parser.parse_args()


if __name__ == '__main__':

    args = _get_parser()

    main(args)
