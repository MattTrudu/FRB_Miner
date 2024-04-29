#!/usr/bin/env python
import os
import sys
import numpy as np
import yaml
import argparse
from bin.launch_heimdall import launch_heimdall
from utils import mkdir_p

def main(args):
    configfile = args.config

    with open(configfile, 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)


    subband_search = config_data['subband_search']

    #RFI
    mask_name = config_data["mask_name"]
    time_start = config_data["time_start"]
    nsamps_gulp = config_data['nsamps_gulp']
    sk_sigma = config_data['sk_sigma']
    sg_sigma = config_data['sg_sigma']
    sg_window = config_data['sg_window']
    plot = config_data['plot']

    #Heimdall

    dm = config_data['dm']
    dm_tolerance = config_data['dm_tolerance']
    boxcar_max = config_data['boxcar_max']
    baseline_length = config_data['baseline_length']
    gpu_id = config_data['gpu_id']
    rfi_no_narrow = config_data['rfi_no_narrow']
    rfi_no_broad = config_data['rfi_no_broad']
    no_scrunching = config_data['no_scrunching']
    scrunching_tol = config_data['scrunching_tol']
    rfi_tol = config_data['rfi_tol']
    nsamps_gulp = config_data['nsamps_gulp']
    fswap = config_data['fswap']

    filename   = args.file
    outdir     = args.output_dir
    pipename   = args.pipeline_name

    filepipeline = os.path.join(outdir, pipename)
    file = open(filepipeline, "w")

    if subband_search == False:
        dirname =  os.path.splitext(os.path.basename(filename))[0]
        #print(dirname)
        outdir = os.path.join(outdir, dirname)
        mkdir_p(outdir)
        rficmd = f"rfi_zapper.py -f {filename} -o {outdir} -n {mask_name} -tstart {time_start} -ngulp {nsamps_gulp} -p {plot} -sksig {sk_sigma} -sgsig {sg_sigma} -sgwin {sg_window}"
        file.write(rficmd+"\n")

        heimdallcmd = launch_heimdall(
        filename,
        dm_tol = dm_tolerance,
        ngulp = nsamps_gulp,
        boxcar_max = boxcar_max,
        baseline_length = baseline_length,
        DM = dm,
        mask = os.path.join(outdir,mask_name)+".bad_chans",
        rfi_no_narrow = rfi_no_narrow,
        rfi_no_broad = rfi_no_broad,
        no_scrunching = no_scrunching,
        rfi_tol = rfi_tol,
        gpu_id = None,
        verbosity = None,
        scrunch_tol = scrunching_tol,
        outdir = outdir,
        fswap = fswap)
        file.write(heimdallcmd+"\n")
    file.close()


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
    parser.add_argument(
        "-n",
        "--pipeline_name",
        help = "Name of the pipeline bash script",
        type = str,
        default = "pipeline.sh",
    )
    return parser.parse_args()


if __name__ == '__main__':

    args = _get_parser()

    main(args)
