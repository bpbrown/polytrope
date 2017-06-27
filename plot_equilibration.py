"""
Plot flux difference and its moving average to determine equilibration. If a single argument is given to <files>... it is interpreted as a directory containing (possibly temporally unjoined) scalar time series. If multiple arguments are given, they are interpreted as a list of temporally-joined scalar timeseries to be further combined. 

Usage:
    plot_equilibration.py [options] <files>...

Options:
    --output=<output>          Output directory; if blank a guess based on likely case name will be made [default: None]
    --rolling=<rolling>        Number of samples in rolling average [default: 100]
    --wallclock                If true, display the wall clock time on the upper x axis
"""
import os
import sys
from docopt import docopt
import pandas as pd
import numpy as np
import h5py
from join_temporal import join_temporal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__.split('.')[-1])


try:
    plt.style.use('modern')
except OSError:
    print("modern style not found. using default.")

def plot_equilibration(sim_time,fep, output_path, wallclock=None,time_boundaries=None):
    flux_equil_pct = pd.Series(fep,index=sim_time)
    rm = []
    for r in rolling:
        rm.append(flux_equil_pct.rolling(r).mean())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(flux_equil_pct.index,flux_equil_pct,alpha=0.3)
    for ir, r in zip(rolling,rm):
        ax.plot(r.index,r,linewidth=3, label="rolling mean over {} samples".format(ir))
    ax.set_xlabel(r"$t/t_{cs}$")
    ax.set_ylabel(r"$\Delta F/F_{bottom}$")
    if len(rolling) > 1:
        ax.legend(loc="upper right")
    if wallclock is not None:
        ax2 = ax.twiny()
        ax2.plot(wallclock[:]/3600.,np.ones(len(wallclock)))
        ax2.cla()
        ax2.set_xlabel("wall clock time (hours)")

    ax.axhline(0,color='k',alpha=0.7)
    ax.set_ylim(-0.5,1.25)
    ax.set_xlim(sim_time[0],sim_time[-1])
    if wallclock is not None:
        ax2.set_xlim(0,wallclock[-1]/3600.)
    plt.tight_layout()

    if time_boundaries is not None:
        for t in time_boundaries:
            ax.axvline(t,alpha=0.4,color='k')
    fig.savefig(os.path.join(output,"flux_equilib.png"),dpi=300)
    return fig

def extract_fep(df):
    fep = df['tasks/flux_equilibration_pct']
    if fep.shape[1] != 1:
        # compatibility with older runs that had mistaken fep diagnostic output
        fep = fep[:,:,0].mean(axis=1)
    else:
        fep = fep[:,0,0]

    return fep

def join_files(filenames,wallclock=False):
    """given a list of filenames, extract sim_time, wallclock (if asked),
    and flux_equil_pct, join them together. Also, return a list of the
    boundaries between different time series.
    """
    sim_time = []
    time_boundaries = []
    fep = []
    if wallclock:
        wc = []
    else:
        wc = None

    wc_offset = 0
    for f in filenames:
        with h5py.File(f,"r") as df:
            sim_time.append(df['scales/sim_time'][:])
            time_boundaries.append(sim_time[-1][-1])
            fep.append(extract_fep(df))
            if wallclock:
                wc.append(df['scales/wall_time'][:]+wc_offset)
                wc_offset = wc[-1][-1]

    sim_time = np.concatenate(sim_time)
    fep = np.concatenate(fep)
    if wallclock:
        wc = np.concatenate(wc)

    return sim_time,fep,wc,time_boundaries[:-1]


if __name__ == "__main__":
    args = docopt(__doc__)

    files = args['<files>']
    rolling = [int(i) for i in args['--rolling'].split(',')]
    wallclock = args['--wallclock']
    wc = None
    if len(files) == 1:
        base = files[0]
        dfname = os.path.join(base,'scalar','scalar_joined.h5')
        try:
            df = h5py.File(dfname,"r")
        except OSError:
            logger.info("Temporally joined scalar file not found. Attempting temporal join...")
            join_temporal(base,data_types=['scalar',])
            df = h5py.File(dfname,"r")
        fep = extract_fep(df)
        if wallclock:
            wc = df['scales/wall_time'][:]
        sim_time = df['scales/sim_time'][:]
        time_boundaries = None
    else:
        base = os.path.commonpath(files)
        sim_time,fep,wc,time_boundaries = join_files(files,wallclock=wallclock)

    output = args['--output']
    if output == 'None':
        output = os.path.join(base,"plots")
    if not os.path.exists(output):
        os.makedirs(output)
    plot_equilibration(sim_time,fep,output,wallclock=wc,time_boundaries=time_boundaries)
