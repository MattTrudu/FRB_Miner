#!/usr/bin/env python
import sys
import os
import argparse
import numpy as np
import matplotlib as mpl
#mpl.use('TkAgg')
import matplotlib.pyplot as plt
import your


def dispersion_delay(fstart, fstop, dms = None):

    """
    Simply computes the delay (in seconds!) due to dispersion. It accepts either a single DM value or a list of DMs. The output will be, respectively, a scalar or a list of delays.
    """

    return (
        4148808.0
        * dms
        * (1 / fstart ** 2 - 1 / fstop ** 2)
        / 1000
    )

def dedisperse(wfall, DM, freq, dt, ref_freq="top"):
    """
    Dedisperse a wfall matrix to DM.
    """

    k_DM = 1. / 2.41e-4
    dedisp = np.zeros_like(wfall)

    # pick reference frequency for dedispersion
    if ref_freq == "top":
        reference_frequency = freq[-1]
    elif ref_freq == "center":
        center_idx = len(freq) // 2
        reference_frequency = freq[center_idx]
    elif ref_freq == "bottom":
        reference_frequency = freq[0]
    else:
        #print "`ref_freq` not recognized, using 'top'"
        reference_frequency = freq[-1]

    shift = (k_DM * DM * (reference_frequency**-2 - freq**-2) / dt).round().astype(int)
    for i,ts in enumerate(wfall):
        dedisp[i] = np.roll(ts, shift[i])
    return dedisp

def renormalize_data(array):

    renorm_data = np.copy(array)
    spec = renorm_data.mean(1)

    renorm_data -= spec[:, np.newaxis]

    baseline = np.mean(renorm_data, axis=0)

    renorm_data -= baseline

    return renorm_data

def plot_data(
    filename,
    output_dir=os.getcwd(),
    output_name="data",
    grab_time=False,
    time_start=None,
    time_stop=None,
    grab_channels=False,
    channel_start=None,
    channel_stop=None,
    save_flag=True,
    file_format=".png",
    renorm_flag = False):

    filedir, name = os.path.split(filename)

    filfile = your.Your(filename)

    dt        = filfile.your_header.native_tsamp
    nsamp     = int(filfile.your_header.native_nspectra)
    nchan     = filfile.your_header.native_nchans
    df        = filfile.your_header.foff
    ftop      = filfile.your_header.fch1
    tstartutc = filfile.your_header.tstart_utc


    fbot = ftop + nchan * df

    channels = np.arange(0, nchan, 1)

    freqs = np.linspace(ftop, fbot, nchan)
    time = np.linspace(0, nsamp * dt, nsamp)

    #data = filterbank.readBlock(0, nsamp)

    if grab_time:
        if time_start < time[0]:
            time_start = time[0]
        if time_stop > time[-1]:
            time_stop = time[-1]

        nstart = int(time_start / dt)
        nstop = int(time_stop / dt)
        ngulp = nstop - nstart
        data = filfile.get_data(nstart = nstart, nsamp = ngulp).T
        time = time[nstart:nstop]
        #data = data[:, nstart : nstop]
    else:
        data = filfile.get_data(nstart = 0, nsamp = nsamp).T


    if grab_channels:
        cstart = int(channel_start)
        cstop  = int(channel_stop)
        if cstart < 0:
            cstart = 0
        if cstop > nchan:
            cstop = nchan
        data = data[cstart:cstop,:]
        channels = channels[cstart:cstop]

    if renorm_flag:
        data = renormalize_data(data)

    timeseries = np.mean(data, axis=0)
    spectrum = np.mean(data, axis=1)

    #np.save("datatest.npy",data)

    fig = plt.figure(figsize=(15, 10))
    mpl.rcParams["axes.linewidth"] = 1.0

    plt.subplots_adjust(top=0.99, bottom=0.1, right=0.99, left=0.1)

    widths = [0.8, 0.2]
    heights = [0.2, 0.8]
    gs = plt.GridSpec(
        2, 2, hspace=0.0, wspace=0, width_ratios=widths, height_ratios=heights
    )

    ax00 = plt.subplot(gs[0, 0])
    ax10 = plt.subplot(gs[1, 0])
    ax11 = plt.subplot(gs[1, 1])

    ax00.set_xticks([])
    ax00.set_yticks([])
    ax11.set_xticks([])
    ax11.set_yticks([])

    size = 15

    ax00.margins(x=0)
    ax11.margins(y=0)
    ax10.tick_params(labelsize=size)
    ax10.tick_params(labelsize=size)
    ax10.set_ylabel("Frequency (MHz)", size=size)
    ax10.set_xlabel("Time (s)", size=size)

    fig.align_labels()

    ax10.imshow(
        data,
        aspect="auto",
        extent=(time[0], time[-1], freqs[channels[-1]], freqs[channels[0]]),
    )
    ax00.plot(time, timeseries, color = "black")
    ax11.plot(np.flipud(spectrum), channels)

    if save_flag:
        output_name = ("%s.%s")%(output_name, file_format)
        plt.savefig(os.path.join(output_dir, output_name))
    else:
        plt.show()


def _get_parser():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Read a SIGPROC filterbank file and plot a portion of the data",
    )
    parser.add_argument(
        "-f",
        "--fil_file",
        action="store",
        help="SIGPROC .fil file to be processed (REQUIRED).",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        action="store",
        help="Output directory (Default: your current path).",
        default="%s/" % (os.getcwd()),
    )
    parser.add_argument(
        "-n",
        "--output_name",
        action="store",
        help="Output File Name (Default: data.png).",
        default=None,
    )
    parser.add_argument(
        "-s",
        "--save_data",
        help="Save the candidate plot.",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--grab_time",
        help="Grab a portion of the data in time (s). Usage -t tstart tstop (Default = False).",
        nargs=2,
        type=float,
        default=None,
    )
    parser.add_argument(
        "-c",
        "--grab_channels",
        help="Grab a portion of the data in frequency channels. Usage -c cstart cstop (Default = False).",
        nargs=2,
        type=int,
        default=None,
    )
    parser.add_argument(
        "-ff",
        "--file_format",
        action="store",
        help="Format of the candidate image (Default: .png).",
        default="png",
    )

    parser.add_argument(
        "-r",
        "--renorm",
        help="Renormalize the data",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = _get_parser()

    filename = args.fil_file
    output_dir = args.output_dir
    output_name = args.output_name
    save_flag = args.save_data
    grab_time = args.grab_time is not None
    time_start, time_stop = args.grab_time or (None, None)
    grab_channels = args.grab_channels is not None
    channel_start, channel_stop = args.grab_channels or (None, None)
    fileformat = args.file_format
    renorm_flag = args.renorm

    plot_data(
        filename,
        output_dir=output_dir,
        output_name=output_name,
        grab_time=grab_time,
        time_start=time_start,
        time_stop=time_stop,
        grab_channels=grab_channels,
        channel_start=channel_start,
        channel_stop=channel_stop,
        save_flag=save_flag,
        file_format=fileformat,
        renorm_flag = renorm_flag)
