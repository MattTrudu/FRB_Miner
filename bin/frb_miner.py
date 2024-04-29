#!/usr/bin/env python
import os
import sys
import numpy as np
import yaml
import argparse
from bin.launch_heimdall import launch_heimdall
from utils import mkdir_p

def main(args):

    filename   = args.file
    configfile = args.config
    outdir     = args.output_dir


    with open(configfile, 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)


    subband_search = config_data['subband_search']
    mask_name = config_data["mask_name"]
    time_start = config_data["time_start"]
    nsamps_gulp = config_data['nsamps_gulp']
    sk_sigma = config_data['sk_sigma']
    sg_sigma = config_data['sg_sigma']
    sg_window = config_data['sg_window']
    plot = config_data['plot']
    if subband_search == False:
        dirname =  os.path.splitext(os.path.basename(filename))[0]
        outdir = mkdir_p(os.path.join(outdir, dirname))
        rficmd = f"rfi_zapper.py -f {filename} -o {outdir} -n {mask_name} -o {outdir} -tstart{time_start} -ngulp {nsamps_gulp} -p {plot} -sksig {sk_sigma} -sgsig {sg_sigma} -sgwin {sg_window}"
        print(rficmd)

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
