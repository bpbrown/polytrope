"""
Plot slices from joint analysis files.

Usage:
    plot_cfl.py join <base_path>
    plot_cfl.py <files>... [options]

Options:
    --output=<output>         Output directory; if blank a guess based on likely case name will be made
    --fields=<fields>         Comma separated list of fields to plot [default: u,w]
    --fast                    Plot "fast" CFL version (using mean z-spacing rather than grid spacing)
"""
import numpy as np
from tools import analysis

import logging
logger = logging.getLogger(__name__.split('.')[-1])

import plot_slices

def main(files, fields, output_path='./', output_name='cfl',
         static_scale=False, fast=False):
    
    from mpi4py import MPI

    comm_world = MPI.COMM_WORLD
    rank = comm_world.rank
    size = comm_world.size

    if fast:
        output_name += '_fast'
    
    data = analysis.Slice(files)

    
    # select down to the data you wish to plot
    data_list = []
    for field in fields:
        logger.info(data.data[field].shape)
        data_list.append(data.data[field][0,:])
    
    dx = np.diff(data.x)
    dx = np.append(dx, dx[-1])
    dx = dx.reshape(dx.shape[0],1)
    if fast:
        # mean spacing over whole sim; not quite one-to-one with the CZ based spacing, but close.
        dz = (data.z[-1]-data.z[0])/data.z.shape[0]
    else:
        dz = np.diff(data.z)
        dz = np.append(dz, dz[-1])
        dz = dz.reshape(1, dz.shape[0])

    frequencies = np.abs(data_list[0])/dx +  np.abs(data_list[1])/dz
    cfl = 1/frequencies
    freq_list=['cfl frequencies', 'cfl times']
    data_list = [frequencies, cfl]
    
    imagestack = plot_slices.ImageStack(data.x, data.z, data_list, freq_list, percent_cut=0)

    scale_late = True
    if static_scale:
        for i, image in enumerate(imagestack.images):
            static_min, static_max = image.get_scale(data_list[i], percent_cut=0.1)
            print(static_min, static_max)
            if scale_late:
                static_min = comm_world.scatter([static_min]*size,root = size-1)
                static_max = comm_world.scatter([static_max]*size,root = size-1)
            else:
                static_min = comm_world.scatter([static_min]*size,root = 0)
                static_max = comm_world.scatter([static_max]*size,root = 0)
            print("post comm: {}--{}".format(static_min, static_max))
            image.set_scale(static_min, static_max)
    imagestack.close()
    
    for i, time in enumerate(data.times):
        current_data = []
        for field in fields:
            current_data.append(data.data[field][i,:])
            
        frequencies = np.abs(current_data[0])/dx +  np.abs(current_data[1])/dz
        cfl = 1/frequencies
        current_data = [frequencies, cfl]

        imagestack = plot_slices.ImageStack(data.x, data.z, current_data, freq_list, verbose=False, percent_cut=0)
        max_freq_loc = np.unravel_index(frequencies.argmax(), frequencies.shape)

        logger.info("cfl limit of {:8.3g}s or {:8.3g}/s at {} : ({:8.3g}, {:8.3g})".format(cfl[max_freq_loc],
                                                                       frequencies[max_freq_loc],
                                                                       max_freq_loc,
                                                                       data.x[max_freq_loc[0]], data.z[max_freq_loc[1]]))
        for image in imagestack.images:
            # Mark where the max frequency occurs (CFL determining location)
            #image.imax.plot([data.x[max_freq_loc[0]], data.z[max_freq_loc[1]]], marker='0', color='green')
            # clumsy tar 
            image.imax.plot(data.x[max_freq_loc[0]], data.z[max_freq_loc[1]],'*y', markeredgecolor='black')
            
        if not static_scale:
            for i_im, image in enumerate(imagestack.images):
                image.set_scale(*image.get_scale(current_data[i_im]))
                                                      
        i_fig = data.writes[i]
        # Update time title
        tstr = 't = {:6.3e}'.format(time)
        imagestack.timestring.set_text(tstr)
        imagestack.write(output_path, output_name, i_fig)
        imagestack.close()


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)
    if args['join']:
        post.merge_analysis(args['<base_path>'])
    else:
        if args['--output'] is not None:
            output_path = pathlib.Path(args['--output']).absolute()
        else:
            data_dir = args['<files>'][0].split('/')[0]
            data_dir += '/'
            output_path = pathlib.Path(data_dir).absolute()
        # Create output directory if needed
        with Sync() as sync:
            if sync.comm.rank == 0:
                if not output_path.exists():
                    output_path.mkdir()
        fields = args['--fields'].split(',')
        logger.info("output to {}".format(output_path))
        
        def accumulate_files(filename,start,count,file_list):
            if start==0:
                file_list.append(filename)
            print(filename, start, count)
        file_list = []
        print(args['<files>'])
        post.visit_writes(args['<files>'],  accumulate_files, file_list=file_list)
            
        if len(file_list) > 0:
            main(file_list, fields, output_path=str(output_path)+'/', fast=args['--fast'])


