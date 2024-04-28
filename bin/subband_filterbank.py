#!/usr/bin/env python
import os
import sys
from sigpyproc.base import Filterbank
from sigpyproc.readers import FilReader
import time
import numpy as np
import argparse




def grab_subband(filename, outdir, outname, chanstart = 0, chanpersub = 1):


    basename = os.path.basename(filename)
    root     = os.path.splitext(basename)[0]

    fil = FilReader("%s"%filename)
    gulp = 1024
    back_compatible = True


    fstart = fil.header.fch1 + chanstart*fil.header.foff
    #print(fstart)
    name = os.path.join(outdir, outname)
    out_file = fil.header.prep_outfile(name,
                                             {"nchans":chanpersub,
                                              "fch1": fstart },
                                              back_compatible=back_compatible, nbits=fil.header.nbits)


    for nsamps, ii, data in fil.read_plan(gulp,verbose = False):
        #for out_file in enumerate(out_files):
            data = data.reshape(nsamps, fil.header.nchans)
            subband_ar = data[:,chanstart:chanstart+chanpersub]
            out_file.cwrite(subband_ar.ravel())


    out_file.close()

    return out_file.name#[out_file.name for out_file in out_files]





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Grab a portion of data from a SIGPROC .fil file and create a sub-banded .fil file")

    parser.add_argument('-f', '--fil_file',   action = "store" , help = "SIGPROC .fil file to be processed", required = True)
    parser.add_argument('-c', '--chan_start',  type = int , action = "store" , help = "Start channel", required = True)
    parser.add_argument('-b', '--band',   type = int , action = "store" , help = "Channels to grab after the start channel", required = True)
    parser.add_argument('-o', '--output_dir', action = "store" , help = "Output for the grabbed filterbank (Default: your current path)", default = "%s"%(os.getcwd()))
    parser.add_argument('-n', '--file_name',  action = "store" , help = "Name for the grabbed filterbank (Default: grabbed_filterbank)", default = "grabbed_filterbank")

    args = parser.parse_args()

    filfile = args.fil_file
    cstart  = args.chan_start
    band    = args.band
    outdir  = args.output_dir
    outname = args.file_name

    grab_subband(filfile, outdir, outname, chanstart = cstart, chanpersub = band)
