#!/usr/bin/env python
import os
import sys
import argparse
import your
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
import getpass
from datetime import datetime
import warnings

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

def renormalize_data(array, badchans_mask = None):

    """
    It takes a spectro-temporal data array and and renormalize the data with respect to a smoothed baseline and make some flagging with an IQR code.
    """


    renorm_data = np.copy(array)
    spec = renorm_data.mean(1)
    if badchans_mask is None:
        badchans_mask = np.zeros(spec.size, dtype=bool)

    for i, newd in enumerate(renorm_data):
        renorm_data[i, :] = (newd - spec[i]) / spec[i]

    baseline = np.mean(renorm_data[~badchans_mask, :], axis=0)
    badbins = np.zeros(baseline.size, dtype=bool)

    if baseline.size > 1000:
        smooth_baseline = gaussian_filter(baseline, 101, truncate = 1)
        detr = baseline - smooth_baseline

        ordered = np.sort(detr)
        q1 = ordered[baseline.size // 4]
        q2 = ordered[baseline.size // 2]
        q3 = ordered[baseline.size // 4 * 3]
        lowlim = q2 - 2 * (q2 - q1)
        hilim = q2 + 2 * (q3 - q2)

        badbins = (detr < lowlim) | (detr > hilim)
        baseline = smooth_baseline

    renorm_data[~badchans_mask, :] -= baseline
    renorm_data[badchans_mask, :] = 0
    renorm_data[:, badbins] = 0

    return renorm_data

def iqr_filter(data, badchans_mask = None):

    """
    Flag noisier channels considering a positive-negative spectrum with a IQR algorithm and add a previous mask.
    """

    newdata = np.copy(data)

    if badchans_mask is None:
        badchans_mask = np.zeros(newdata.size, dtype=bool)

    spec = np.std(newdata, axis=1)

    ordered = np.sort(spec[~badchans_mask])

    q1 = ordered[ordered.size // 4]
    q2 = ordered[ordered.size // 2]
    q3 = ordered[ordered.size // 4 * 3]
    lowlim = q2 - 2 * (q2 - q1)
    hilim = q2 + 2 * (q3 - q2)

    badchans = (spec < lowlim) | (spec > hilim) | badchans_mask

    return badchans

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

def prepare_data(data, badchans, freqs, tsamp, DM):

    data = np.asarray(data , dtype = np.float32)

    data = renormalize_data(data, badchans_mask = badchans)

    badchans = iqr_filter(data, badchans_mask = badchans) | badchans

    data[badchans, :] = 0


    dedispdata = dedisperse(data, DM, freqs, tsamp, ref_freq="bottom")

    dedispdata[badchans, :] = 0


    return badchans, dedispdata

def downsample_mask(badchans, newshape):

    """
    Downsample an RFI mask. It considers, as way of downsampling, group of channels and from them it takes as a new downsampled value the most common value between 0 or 1.
    """

    oldshape = badchans.shape[0]

    ratio = int(oldshape / newshape)

    downbadchans = np.zeros((newshape,), dtype = int)

    badchansn = np.zeros((oldshape,) , dtype = int)
    badchansn[badchans] = 1

    for k in range(newshape):

        values, counts = np.unique(badchansn[k * ratio : (k+1) * ratio], return_counts = True)
        ind = np.argmax(counts)
        downbadchans[k] = values[ind]

    downbadchans = np.asarray(downbadchans, dtype = bool)
    return downbadchans

def DMT(dedispdata, freqs, dt, DM = 0, dmsteps = 1024):

    dmrange = 0.25 * DM

    DMs = np.linspace(-dmrange, dmrange, dmsteps)

    dmt = np.zeros((dmsteps, dedispdata.shape[1]))

    for k,dm in enumerate(DMs):

        data = dedisperse(dedispdata, dm, freqs, dt, ref_freq="bottom")
        dmt[k,:] = data.mean(0)

    return dmt

def plot_candidate(filename,
    tcand_s,
    boxcar = 0,
    dm = 0,
    window_ms = 100,
    sk_sig = 5,
    sg_sig = 5,
    sg_win = 41,
    verbose = False,
    grab_channels = False,
    cstart = None,
    cstop = None,
    fshape = 256,
    tshape = 256,
    save = False,
    outname = "candidate",
    output_dir = os.getcwd()):

    warnings.filterwarnings("ignore")

    filedir, name = os.path.split(filename)

    filfile = your.Your(filename)

    dt        = filfile.your_header.native_tsamp
    nsamp     = int(filfile.your_header.native_nspectra)
    nchan     = filfile.your_header.native_nchans
    df        = filfile.your_header.foff
    ftop      = filfile.your_header.fch1
    tstartutc = filfile.your_header.tstart_utc
    fbot = ftop + nchan * df



    freqs  = np.linspace(ftop, fbot, nchan)

    ddelay = dispersion_delay(fbot, ftop, dms = dm)

    ncand  = int(tcand_s / dt)
    ndelay = int(ddelay / dt)
    if boxcar == 0:
        wing = 1
    else:
        wing = int(2**(boxcar -1))

    if verbose:
        print("Grabbing data...")

    data = filfile.get_data(nstart = ncand - ndelay - wing, nsamp = 2 * (ndelay + wing) ).T

    if verbose:
        print("SK flagging...")
    startmask = your.utils.rfi.sk_sg_filter(data[:,0:1024].T, filfile, sk_sig, sg_win, sg_sig)

    if verbose:
        print("Dedisperding...")

    badchans, dedispdata = prepare_data(data, startmask, freqs, dt, dm)

    if verbose:
        print(f"Taking {window_ms} ms around the candidate...")

    data = data[:, ndelay - wing : -1]
    data[badchans,:] = 0

    window_s = window_ms * 1e-3
    window_bin =  int(window_s / dt)

    dedispdata = dedispdata[:, ndelay + wing - window_bin : ndelay + wing + window_bin]

    if grab_channels:
        if verbose:
            print("Grabbing the selected sub-band...")
        cstart = int(cstart)
        cstop  = int(cstop)
        dedispdata = dedispdata[cstart : cstop,:]
        data = data[cstart : cstop, :]
        freqs = freqs[cstart : cstop]

    if verbose:
        print("Preparing DM-time plot...")

    if grab_channels:
        dmt = DMT(dedispdata[cstart : cstop, :], freqs, dt, DM = dm)
    else:
        dmt = DMT(dedispdata, freqs, dt, DM = dm)

    if verbose:
        print("Resizing...")

    if tshape <= dmt.shape[1]:
        dmt = resize(dmt, (dmt.shape[0], tshape), anti_aliasing = True)
    if tshape <= data.shape[1]:
        data = resize(data, (data.shape[0], tshape), anti_aliasing = True)
    if fshape <= data.shape[0]:
        data = resize(data, (fshape, data.shape[1]), anti_aliasing = True)
    if tshape <= dedispdata.shape[1]:
        dedispdata = resize(dedispdata, (dedispdata.shape[0], tshape), anti_aliasing = True)
    if fshape <= dedispdata.shape[0]:
        dedispdata = resize(dedispdata, (fshape, dedispdata.shape[1]), anti_aliasing = True)

    if verbose:
        print("Plotting...")

    plt.style.use('dark_background')
    figure = plt.figure(figsize = (10,7))
    size = 12

    widths0  = [0.8,0.2]
    widths1  = [1]
    heights0 = [0.2,0.4,0.4]
    heights1 = [0.2,0.4,0.4]


    gs0  = plt.GridSpec(3,2,hspace = 0.0 , wspace = 0,  width_ratios = widths0, height_ratios = heights0, top = 0.99 , bottom = 0.1, right = 0.55, left = 0.10)
    gs1  = plt.GridSpec(3,1,hspace = 0.0 , wspace = 0,  width_ratios = widths1, height_ratios = heights1, top = 0.99 , bottom = 0.1, right = 0.99, left = 0.65)

    ax0_00 = plt.subplot(gs0[0,0])
    ax0_10 = plt.subplot(gs0[1,0])
    #ax0_11 = plt.subplot(gs0[1,1])
    ax0_20 = plt.subplot(gs0[2,0])
    ax1_20 = plt.subplot(gs1[2,0])

    size = 15
    ax0_00.set_xticks([])
    #ax0_00.set_yticks([])
    ax0_10.set_xticks([])
    #ax0_11.set_xticks([])
    #ax0_11.set_yticks([])
    ax0_00.margins(x=0)
    #ax0_11.margins(y=0)

    ax0_00.set_ylabel(r"$\sigma$", size = size)

    ax0_10.set_ylabel("Frequency (MHz)", size = size)
    ax0_20.set_ylabel(r"DM (pc$\times$cm$^{-3}$)", size = size)
    ax0_20.set_xlabel("Time (ms)", size = size)

    ax1_20.set_ylabel("Frequency (MHz)", size = size)
    ax1_20.set_xlabel("Time (s)", size = size)

    #ax0_00.spines['top'].set_color('k')
    #ax0_00.spines['right'].set_color('k')
    #ax0_00.spines['left'].set_color('k')
    #ax0_00.spines['bottom'].set_color('k')

    lc = np.mean(dedispdata, axis = 0)
    mask = np.ones(lc.shape[0], dtype = bool)
    amax = np.argmax(lc)
    mask[amax - wing : amax + wing ] = 0
    mu  = np.mean(lc[~mask]) # mean off-burst
    std = np.std(lc[~mask])  # rms off-burst
    lc = (lc - mu) / std

    ax0_00.step(np.arange(lc.shape[0]), lc, linewidth = 1, color = "magenta")
    T = 0.5 * window_ms #+ wing * dt * 1e3
    ax0_10.imshow(dedispdata, aspect = "auto", cmap = "inferno", extent = (-T,T,freqs[-1],freqs[0]))
    ax0_20.imshow(dmt, aspect = "auto", extent = (-T,T, dm + 0.25 * dm, dm - 0.25 * dm))

    ax1_20.imshow(data, aspect = "auto", extent = (0, ddelay + wing * dt, freqs[-1], freqs[0]), cmap = "inferno")

    username = getpass.getuser()
    datetimenow = datetime.utcnow()
    figure.text(0.650,0.01,"Plot made by %s on %s UTC"%(username,str(datetimenow)[0:19]), fontsize = 8)

    figure.text(0.650,0.700, f"File name: {name}" ,fontsize = 10)


    figure.text(0.650,0.800, f"Candidate arrival time (s) = {tcand_s}" ,fontsize = 10)
    figure.text(0.650,0.775, r"Candidate DM (pc$\times$cm$^{-3}$) = " + f"{dm}" ,fontsize = 10)
    figure.text(0.650,0.750, f"Candidate Box-car width (ms) = {(wing * dt * 1e3):.2f}" ,fontsize = 10)
    plt.tight_layout()


    if save:
        output_name = f"{outname}.png"
        plt.savefig(os.path.join(output_dir, output_name))
    else:
        plt.show()

def _get_parser():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Read a SIGPROC filterbank file and plot an FRB candidate.",
    )
    parser.add_argument(
        "-f",
        "--file",
        action = "store",
        help = "SIGPROC .fil file to be processed (REQUIRED).",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--time_cand",
        type = float,
        help = "Arrival time of the candidate in seconds.",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--dm_cand",
        type = float,
        help = "Dispersion measure of the candidate in pc cm^-3.",
        default = 0,
    )
    parser.add_argument(
        "-b",
        "--box_car",
        type = int,
        default = 0,
        help = "Box-car width in power of 2 of the candidate",
    )
    parser.add_argument(
        "-tw",
        "--time_window",
        type = float,
        help = "Time window to grab and plot around the dedispersed burst (Default: 100 ms)",
        default = 100,
    )

    parser.add_argument(
        "-sksig",
        "--sk_sigma",
        help = "Spectral Kurtosis sigma",
        type = int,
        default = 5,
    )

    parser.add_argument(
        "-sgsig",
        "--sg_sigma",
        help = "Savitzky-Golay sigma",
        type = int,
        default = 5,
    )

    parser.add_argument(
        "-sgwin",
        "--sg_window",
        help = "Savitzky-Golay window (in MHz)",
        type = int,
        default = 41,
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action = "store_true",
        help = "Be verbose",
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
        help="Output File Name (Default: candidate.png).",
        default="candidate",
    )
    parser.add_argument(
        "-s",
        "--save_data",
        help="Save the candidate plot.",
        action="store_true",
    )
    parser.add_argument('-fs',
                        '--f_shape',
                        type = int,
                        default = None,
                        action = "store" ,
                        help = "Shape of the data in frequency"
                        )
    parser.add_argument('-ts',
                        '--t_shape',
                        type = int,
                        default = None,
                        action = "store" ,
                        help = "Shape of the data in time"
                        )
    parser.add_argument(
                        "-c",
                        "--grab_channels",
                        help="Grab a portion of the data in frequency channels. Usage -c cstart cstop (Default = False).",
                        nargs=2,
                        type=int,
                        default=None,
                        )


    return parser.parse_args()

if __name__ == '__main__':

    args = _get_parser()

    gchan = args.grab_channels is not None
    chanstart, chanstop = args.grab_channels or (None, None)
    plot_candidate(args.file,
        args.time_cand,
        boxcar = args.box_car,
        dm = args.dm_cand,
        window_ms = args.time_window,
        sk_sig = args.sk_sigma,
        sg_sig = args.sg_sigma,
        sg_win = args.sg_window,
        verbose = args.verbose,
        grab_channels = gchan,
        cstart = chanstart,
        cstop = chanstop,
        fshape = args.f_shape,
        tshape = args.t_shape,
        save = args.save_data,
        outname = args.output_name,
        output_dir = args.output_dir)
