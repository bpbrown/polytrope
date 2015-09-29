"""
Plot slices from joint analysis files.

Usage:
    plot_slices.py join <base_path>
    plot_slices.py <files>... [--fields=<fields> --output=<output>]

Options:
    --output=<output>  Output directory; if blank a guess based on likely case name will be made
    --fields=<fields>  Comma separated list of fields to plot [default: s',enstrophy]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import brewer2mpl
from mpi4py import MPI

import analysis

import logging
logger = logging.getLogger(__name__.split('.')[-1])

comm_world = MPI.COMM_WORLD
rank = comm_world.rank
size = comm_world.size

class Colortable():
    def __init__(self, field,
                 reverse_scale=True, float_scale=False, logscale=False):
        
        if field=='enstrophy':
            self.color_map = ('BuPu', 'sequential', 9)
        else:
            self.color_map = ('RdYlBu', 'diverging', 11)
            
        self.reverse_scale = reverse_scale
        self.float_scale = float_scale
        self.logscale = logscale

        self.cmap = brewer2mpl.get_map(*self.color_map, reverse=self.reverse_scale).mpl_colormap

class ImageStack():
    def __init__(self, x, y, fields, field_names, true_aspect_ratio=True, vertical_stack=True, scale=3.0):
        # Storage
        images = []
        image_axes = []
        cbar_axes = []

        # Determine grid size
        if vertical_stack:
            nrows = len(fields)
            ncols = 1
        else:
            nrows = 1
            ncols = len(fields)

        # Setup spacing [top, bottom, left, right] and [height, width]
        t_mar, b_mar, l_mar, r_mar = (0.2, 0.2, 0.2, 0.2)
        t_pad, b_pad, l_pad, r_pad = (0.15, 0.03, 0.03, 0.03)
        h_cbar, w_cbar = (0.05, 1.)

        domain_width = np.max(x)-np.min(x)
        domain_height = np.max(y)-np.min(y)
        if true_aspect_ratio:
          h_data, w_data = (1., domain_width/domain_height)
        else:
          h_data, w_data = (1., 1.)

        h_im = t_pad + h_cbar + h_data + b_pad
        w_im = l_pad + w_data + r_pad
        h_total = t_mar + nrows * h_im + b_mar
        w_total = l_mar + ncols * w_im + r_mar

        self.dpi_png = int(max(150, len(x)/(w_total*scale)))
        logger.info("figure size is {:g}x{:g} at {} dpi".format(scale * w_total, scale * h_total, self.dpi_png))
        
        # Create figure and axes
        self.fig = fig = plt.figure(1, figsize=(scale * w_total,
                                                scale * h_total))
        row = 0
        cindex = 0

        for j, field in enumerate(fields):
            field_name = field_names[j]
            
            left = (l_mar + w_im * cindex + l_pad) / w_total
            bottom = 1 - (t_mar + h_im * (row + 1) - b_pad) / h_total
            width = w_data / w_total
            height = h_data / h_total
            imax = fig.add_axes([left, bottom, width, height])
            image_axes.append(imax)
            image_axes[j].lastrow = (row == nrows - 1)
            image_axes[j].firstcol = (cindex == 0)

            left = (l_mar + w_im * cindex + l_pad) / w_total
            bottom = 1 - (t_mar + h_im * row + t_pad + h_cbar) / h_total
            width = w_cbar / w_total
            height = h_cbar / h_total
            cbax = fig.add_axes([left, bottom, width, height])
            cbar_axes.append(cbax)

            cindex+=1
            if cindex%ncols == 0:
                # wrap around and start the next row
                row += 1
                cindex = 0

            image = Image(field_name,imax,cbax)
            image.add_image(fig,x,y,field[0].T)
            images.append(image)
            
        # Title
        height = 1 - (0.6 * t_mar) / h_total
        self.timestring = fig.suptitle(r'', y=height, size=16)
        # Set up images and labels        
        

        #

        ## if static_scale:
        ##     if fname in log_list:
        ##         static_min, static_max = set_scale(field, even_scale=False, percent_cut=[0.4, 0.0])
        ##     else:
        ##         # center on zero
        ##         static_min, static_max = set_scale(field, even_scale=even_scale, percent_cut=0.1)

        ##     if scale_late:
        ##         static_min = comm_world.scatter([static_min]*size,root = size-1)
        ##         static_max = comm_world.scatter([static_max]*size,root = size-1)
        ##     else:
        ##         static_min = comm_world.scatter([static_min]*size,root = 0)
        ##         static_max = comm_world.scatter([static_max]*size,root = 0)

        ##     images[j].set_clim(static_min, static_max)
        ##     print(fname, ": +- ", -static_min, static_max)


    def write(self, data_dir, name, i_fig):
        figure_file = "{:s}/{:s}_{:06d}.png".format(data_dir,name,i_fig)
        self.fig.savefig(figure_file, dpi=self.dpi_png)
        print("writting {:s}".format(figure_file))
            
class Image():
    def __init__(self, field_name, imax, cbax, float_scale=False, fixed_lim=None, even_scale=True, units=True):

        self.xstr = 'x/H'
        self.ystr = 'z/H'

        self.imax = imax
        self.cbax = cbax
        
        self.colortable = Colortable(field_name)
        self.field_name = field_name
        self.float_scale = float_scale
        self.fixed_lim = fixed_lim
        self.even_scale = even_scale
        self.units = units
        self.add_labels(field_name)
        
    def add_labels(self, fname):
        imax = self.imax
        cbax = self.cbax
        
        # Title
        title = imax.set_title('{:s}'.format(fname), size=14)
        title.set_y(1.1)

        # Colorbar
        self.cbax.xaxis.set_ticks_position('top')
        plt.setp(cbax.get_xticklabels(), size=10)

        if imax.lastrow:
            imax.set_xlabel(self.xstr, size=12)
            plt.setp(imax.get_xticklabels(), size=10)
        else:
            plt.setp(imax.get_xticklabels(), visible=False)

        if imax.firstcol:
            self.imax.set_ylabel(self.ystr, size=12)
            plt.setp(imax.get_yticklabels(), size=10)
        else:
            plt.setp(imax.get_yticklabels(), visible=False)
             
    def create_limits_mesh(self, x, y):
        xd = np.diff(x)
        yd = np.diff(y)
        shape = x.shape
        xm = np.zeros((y.size+1, x.size+1))
        ym = np.zeros((y.size+1, x.size+1))
        xm[:, 0] = x[0] - xd[0] / 2.
        xm[:, 1:-1] = x[:-1] + xd / 2.
        xm[:, -1] = x[-1] + xd[-1] / 2.
        ym[0, :] = y[0] - yd[0] / 2.
        ym[1:-1, :] = (y[:-1] + yd / 2.)[:, None]
        ym[-1, :] = y[-1] + yd[-1] / 2.

        return xm, ym
           
    def add_image(self, fig, x, y, data):
        imax = self.imax
        cbax = self.cbax
        cmap = self.colortable.cmap
        
        if self.units:
            xm, ym = self.create_limits_mesh(x, y)

            im = imax.pcolormesh(xm, ym, data, cmap=cmap, zorder=1)
            plot_extent = [xm.min(), xm.max(), ym.min(), ym.max()]
            imax.axis(plot_extent)
        else:
            im = imax.imshow(data, zorder=1, aspect='auto',
                             interpolation='none', origin='lower',
                             cmap=cmap)
            shape = data.shape
            plot_extent = [-0.5, shape[1] - 0.5, -0.5, shape[0] - 0.5]
            imax.axis(plot_extent)

        cb = fig.colorbar(im, cax=cbax, orientation='horizontal',
                          ticks=ticker.MaxNLocator(nbins=5, prune='both'))

        cb.formatter.set_powerlimits((4, 3))
        cb.update_ticks()
        self.im = im
    
    def update_image(self, data):
        im = self.im
        
        if self.units:
            im.set_array(np.ravel(data))
        else:
            im.set_data(data)

        if not self.static_scale or self.float_scale:
            image_min, image_max = self.set_scale(field, fixed_lim=fixed_lim, even_scale=even_scale)
            self.im.set_clim(image_min, image_max)

    def percent_trim(self, field, percent_cut=0.03):
        if isinstance(percent_cut, list):
            if len(percent_cut) > 1:
                low_percent_cut  = percent_cut[0]
                high_percent_cut = percent_cut[1]
            else:
                low_percent_cut  = percent_cut[0]
                high_percent_cut = percent_cut[0]
        else:
            low_percent_cut  = percent_cut
            high_percent_cut = percent_cut

        # trimming method from Ben's ASH analysis package
        sorted_field = np.sort(field, axis=None)
        N_elements = len(sorted_field)
        min_value = sorted_field[low_percent_cut*N_elements]
        max_value = sorted_field[(1-high_percent_cut)*N_elements-1]
        return min_value, max_value

    def set_scale(field, fixed_lim=None, even_scale=True, percent_cut=0.03):
        if fixed_lim is None:
            if even_scale:
                image_min, image_max = self.percent_trim(field, percent_cut=percent_cut)
                if np.abs(image_min) > image_max:
                    image_max = np.abs(image_min)
                elif image_min < 0:
                    image_min = -np.abs(image_max)
            else:
                image_min, image_max = self.percent_trim(field, percent_cut=percent_cut)
        else:
            image_min = fixed_lim[0]
            image_max = fixed_lim[1]

        return image_min, image_max

    

def main(files, fields, output_path='./', output_name='snapshot'):
    data = analysis.Slice(files)
    field_data = []
    for field in fields:
        field_data.append(data.data[field])
    
    imagestack = ImageStack(data.x, data.z, field_data, fields)

    for i, time in enumerate(data.times):
        i_fig = data.writes[i]
        logger.info(i_fig)
        # Update time title
        tstr = 't = {:6.3e}'.format(time)
        imagestack.timestring.set_text(tstr)
        imagestack.write(output_path, output_name, i_fig)
            


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
        main(args['<files>'], fields, output_path=str(output_path)+'/')


