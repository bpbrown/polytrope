"""
Plot slices from joint analysis files.

Usage:
    plot_plume_overshoot.py join <base_path>
    plot_plume_overshoot.py <files>... [options]

Options:
    --output=<output>         Output directory; if blank a guess based on likely case name will be made
    --fields=<fields>         Comma separated list of fields to plot [default: s',enstrophy]
    --profiles=<profiles>...  Files to use for contour plotting
"""
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import brewer2mpl

import analysis

import logging
logger = logging.getLogger(__name__.split('.')[-1])

import plot_slices

def main(files, fields, output_path='./', output_name='plume', static_scale=True, profile_files=None):
    from mpi4py import MPI

    comm_world = MPI.COMM_WORLD
    rank = comm_world.rank
    size = comm_world.size
    
    data = analysis.Slice(files)
    
    # select down to the data you wish to plot
    data_list = []
    for field in fields:
        logger.info(data.data[field].shape)
        data_list.append(data.data[field][0,:])
        
    imagestack = plot_slices.ImageStack(data.x, data.z, data_list, fields)

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

    if profile_files is not None:
        logger.info(profile_files)
        profile_data = analysis.Profile(profile_files)
              
    for i, time in enumerate(data.times):
        current_data = []
        for field in fields:
            current_data.append(data.data[field][i,:])

        # new ImageStack on each image, to clear out all axes when doing contouring.
        imagestack = plot_slices.ImageStack(data.x, data.z, current_data, fields, verbose=False)
        zoom = True
        if zoom:
            for image in imagestack.images:
                image.set_limits([0,0.25*max(data.x)], [0.25*max(data.z), 0.5*max(data.z)])
        
        contour = True
        if contour:
            s_mean = profile_data.average['s_mean']
            s_mean = s_mean.reshape(s_mean.shape[0], 1)
            contour_data = data.data['s'][i,:].T + s_mean
            
            for j, field in enumerate(fields):
                imagestack.images[j].imax.contour(data.x, data.z, contour_data,
                                                  levels=[0], colors='black', linewidths=3,
                                                  antialiased=True)
                                              
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
        if args['--profiles'] is not None:
            #profile_file_list = []
            #print(args['--profiles'])
            #post.visit_writes(args['--profiles'],  accumulate_files, file_list=profile_file_list)
            #print(profile_file_list)
            profile_file_list = [args['--profiles']]
        else:
            profile_file_list = None
            
        if len(file_list) > 0:
            main(file_list, fields, output_path=str(output_path)+'/', profile_files=profile_file_list)


