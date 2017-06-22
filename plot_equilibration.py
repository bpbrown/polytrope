"""
Plot flux difference and its moving average to determine equilibration

Usage:
    plot_equilibration.py [options] <files>

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
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__.split('.')[-1])


try:
    plt.style.use('modern')
except OSError:
    print("modern style not found. using default.")

if __name__ == "__main__":
    args = docopt(__doc__)

    base = args['<files>']
    rolling = [int(i) for i in args['--rolling'].split(',')]
    output = args['--output']
    wallclock = args['--wallclock']

    if output == 'None':
        output = os.path.join(base,"plots")
        if not os.path.exists(output):
            os.makedirs(output)

    dfname = os.path.join(base,'scalar','scalar_joined.h5')
    try:
        df = h5py.File(dfname,"r")
    except OSError:
        logger.info("Temporally joined scalar file not found. Attempting temporal join...")
        join_temporal(base,data_types=['scalar',])
        df = h5py.File(dfname,"r")

    fep = df['tasks/flux_equilibration_pct']
    if fep.shape[1] != 1:
        # compatibility with older runs that had mistaken fep diagnostic output
        fep = fep[:,:,0].mean(axis=1)
    else:
        fep = fep[:,0,0]
    
    flux_equil_pct = pd.Series(fep,index=df['scales/sim_time'][:])
    rm = []
    for r in rolling:
        rm.append(flux_equil_pct.rolling(r).mean())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if wallclock:
        ax2 = ax.twiny()
        ax2.plot(df['scales/wall_time'][:]/3600.,np.ones(len(df['scales/wall_time'])))
        ax2.cla()
        ax2.set_xlabel("wall clock time (hours)")
    ax.plot(flux_equil_pct.index,flux_equil_pct,alpha=0.3)
    for ir, r in zip(rolling,rm):
        ax.plot(r.index,r,linewidth=3, label="rolling mean over {} samples".format(ir))
    ax.set_xlabel(r"$t/t_{cs}$")
    ax.set_ylabel(r"$\Delta F/F_{bottom}$")
    if len(rolling) > 1:
        ax.legend(loc="upper right")
    ax.axhline(0,color='k',alpha=0.7)
    ax.set_ylim(-1.25,1.25)
    plt.tight_layout()
    fig.savefig(os.path.join(output,"flux_equilib.png"),dpi=300)
