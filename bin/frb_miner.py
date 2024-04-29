#!/usr/bin/env python
import os
import sys
import numpy as np
import yaml
import argparse
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

    #prepare_for_fetch
    snr = config_data['snr']
    dmf =  config_data['dmf']
    n_members =  config_data['n_members']

    #fetch
    probability = config_data['probability']
    model = config_data['model']

    filename   = args.file
    outdir     = args.output_dir
    pipename   = args.pipeline_name
    slurm      = args.slurm

    filepipeline = os.path.join(outdir, pipename)
    file = open(filepipeline, "w")

    if slurm:
        n_nodes = config_data['n_nodes']
        n_cpu = config_data['n_cpu']
        computing_time = config_data['computing_time']
        file.write(f"#!/bin/bash\n")
        file.write(f"#SBATCH --job-name = {pipeline.replace(".sh","")}\n")
        file.write(f"#SBATCH --nodes={n_nodes}\n")
        file.write(f"#SBATCH --ntasks-per-node=1\n")
        file.write(f"#SBATCH --gres=gpu:1\n")
        file.write(f"#SBATCH --cpus-per-task={n_cpu}\n")
        file.write(f"#SBATCH --time={computing_time}\n")

        file.write("module purge\n")
        file.write("module load python/3.8.13")
        file.write("module load pulsar/heimdallGPU") 

    if subband_search == False:
        dirname =  os.path.splitext(os.path.basename(filename))[0]
        #print(dirname)
        outdir = os.path.join(outdir, dirname)
        mkdir_p(outdir)
        rficmd = f"rfi_zapper.py -f {filename} -o {outdir} -n {mask_name} -tstart {time_start} -ngulp {nsamps_gulp} -p {plot} -sksig {sk_sigma} -sgsig {sg_sigma} -sgwin {sg_window}"
        file.write(rficmd+"\n")
        maskpath = os.path.join(outdir,mask_name)+".bad_chans"
        heimdallcmd = f"launch_heimdall.py -f {filename} -o {outdir} -dm {dm[0]} {dm[1]} -m {maskpath} -box_max {boxcar_max} -dm_tol {dm_tolerance} -ngulp = {nsamps_gulp} -fswap {fswap} -base_len {baseline_length} -rfi_no_narrow {rfi_no_narrow} -rfi_no_broad {rfi_no_broad} -no_scrunching {no_scrunching} -rfi_tol {rfi_tol} -scrunch_tol {scrunching_tol}"
        file.write(heimdallcmd+"\n")
        prepcmd = f"prepare_for_fetch.py -f {filename} -m {maskpath} -o {outdir} -c {outdir} -d {dmf[0]} {dmf[1]} -s {snr} -n {n_members}"
        file.write(prepcmd+"\n")
        cand = os.path.join(outdir,"*.cand")
        file.write(f"rm -f {cand}"+"\n")
        csvpath = os.path.join(outdir, "cand_forfetch.csv")
        candmakercmd = f"your_candmaker.py -c {csvpath} -o {outdir}"
        file.write(candmakercmd + "\n")
        fetchcmd = f"predict.py -c {outdir} -m {model} -p {probability}"
        file.write(fetchcmd+"\n")
        results = os.path.join(outdir, f"results_{model}.csv" )
        ploth5cmd = f"your_h5plotter.py -c {results} -o {outdir}"
        file.write(ploth5cmd+"\n")
        file.write(f"rm -f {outdir}/*.h5 \n")

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
        '-s',
        '--slurm',
        action = "store_false",
        help = "Make a SLURM job rather than a bash script",
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
