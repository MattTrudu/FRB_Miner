#!/usr/bin/env python
import numpy as np
import os
import argparse
import csv

def make_for_fetch(filfile, canddir, outdir, mask = " ", dms = [0,1000], snr = 6, Nmember = 20):

    cand_files = [file for file in os.listdir(canddir) if file.endswith(".cand")]

    filename = f"cand_forfetch.csv"
    filename = os.path.join(outdir, filename)
    filename_full = f"heimdall_result.csv"
    filename_full = os.path.join(outdir, filename_full)

    filecsv = open(filename, "w")
    heimcsv = open(filename_full, "w")
    filecsv.write("file,snr,stime,width,dm,label,chan_mask_path,num_files\n")
    heimcsv.write("snr,stime,width,dm,members\n")
# Loop through each .cand file

    for fname in cand_files:
        file_path = os.path.join(canddir, fname)

        # Open .cand file and extract specified columns
        with open(file_path, 'r') as cand_file:
            for line in cand_file:
                # Split the line by spaces
                v = line.split()
                sn, toa, boxcar, dm, members = float(v[0]), float(v[2]), int(v[3]), float(v[5]), int(v[6])
                stringf = f"{sn},{toa},{boxcar},{dm},{members}\n"
                heimcsv.write(stringf)
                if sn >= snr and dms[0] <= dm <= dms[1] and members <= Nmember:
                    string = f"{filfile},{snr},{toa},{boxcar},{dm},0,{mask},1\n"
                    #"%s,%.5f,%.5f,%0d,%.5f,0,%s,1\n"%(filfile, snr[lineidx], tcand[lineidx], filter[lineidx], dm[lineidx], mask
                    filecsv.write(string)

    filecsv.close()
    heimcsv.close()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Create a .csv file ready for FETCH from an Heimdall candidate files. It also allows to filter according to S/N and DM and Nmembers")

    parser.add_argument('-f', '--fil_file',   action = "store" ,  help = "Path of the SIGPROC .fil file", required = True)
    parser.add_argument('-m', '--mask_file',   action = "store" , help = "Path of the mask file", default = " ")
    parser.add_argument('-c', '--cand_dir',   action = "store" , help = "Path of the candidate files obtained from Heimdall", required = True)
    parser.add_argument('-o', '--output_dir', action = "store" , help = "Output for FETCH (Default: your current path)", default = "%s/"%(os.getcwd()))
    parser.add_argument("-d","--dm", help = "Dispersion Measure (pc cm^-3) to filter between (Default: 0 1000)",type = float,nargs = 2, default = [0,1000])
    parser.add_argument("-s","--snr", action = "store", type = float, default = 6, help = "S/N threshold to filter")
    parser.add_argument("-n","--n_members", action = "store", type = int, default = 3, help = "Maximum number of members (DM/boxcar clustered points) allowed to not be considered noise")

    args = parser.parse_args()

    filfile  = args.fil_file
    canddir  = args.cand_dir
    mask     = args.mask_file
    outdir   = args.output_dir
    snr      = args.snr
    dms      = args.dm
    nmemb    = args.n_members

    make_for_fetch(filfile, canddir, outdir, mask = mask, dms = dms, snr = snr, Nmember = nmemb)
