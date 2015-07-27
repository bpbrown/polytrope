import numpy as np
import h5py

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from collections import OrderedDict

class DedalusData():
    def __init__(self,  files, *args,
                 keys=None, verbose=False, **kwargs):
        
        self.verbose = verbose

        self.files = sorted(files, key=lambda x: int(x.split('.')[0].split('_s')[1]))
        logger.debug("opening: {}".format(self.files))
        
        if keys is None:
            self.get_keys(self.files[0], keys=keys)
        else:
            self.keys = keys
            
        self.data = OrderedDict()
        for key in self.keys:
            self.data[key] = np.array([])

        if self.verbose:
            self.report_contents(self.files[0])
            
    def get_keys(self, file, keys=None):
        f = h5py.File(file, flag='r')
        self.keys = np.copy(f['tasks'])
        f.close()
        logger.debug("tasks to study = {}".format(self.keys))

    def report_contents(self, file):
        f = h5py.File(file, flag='r')
        logger.info("Contents of {}".format(file))
        logger.info(10*'-'+' tasks '+10*'-')
        for task in f['tasks']:
            logger.info(task)
        
        logger.info(10*'-'+' scales '+10*'-')
        for key in f['scales']:
            logger.info(key)
        f.close()
        
class Scalar(DedalusData):
    def __init__(self, files, *args, keys=None, **kwargs):
        super(Scalar, self).__init__(files, *args,
                                     keys=keys, **kwargs)
        self.read_data()
            
    def read_data(self):
        self.times = np.array([])
        
        N = 1
        for filename in self.files:
            logger.debug("opening {}".format(filename))
            f = h5py.File(filename, flag='r')
            # clumsy
            for key in self.keys:
                if N == 1:
                    self.data[key] = f['tasks'][key][:]
                    logger.debug("{} shape {}".format(key, self.data[key].shape))
                else:
                    self.data[key] = np.append(self.data[key], f['tasks'][key][:], axis=0)

            N += 1
            self.times = np.append(self.times, f['scales']['sim_time'][:])
            f.close()
            
        for key in self.keys:
            self.data[key] = self.data[key][:,0,0]
            logger.debug("{} shape {}".format(key, self.data[key].shape))
            
class Profile(DedalusData):
    def __init__(self, files, *args, keys=None, **kwargs):
        super(Profile, self).__init__(files, *args,
                                      keys=keys, **kwargs)
        self.read_data()
        self.average_data()
        
    def read_data(self):

        self.times = np.array([])

        N = 1
        for filename in self.files:
            f = h5py.File(filename, flag='r')
            # clumsy
            for key in self.keys:
                if N == 1:
                    self.data[key] = f['tasks'][key][:]
                    logger.debug("{} shape {}".format(key, self.data[key].shape))
                else:
                    self.data[key] = np.append(self.data[key], f['tasks'][key][:], axis=0)

            N += 1
            # same z for all files
            self.z = f['scales']['z']['1.0'][:]
            self.times = np.append(self.times, f['scales']['sim_time'][:])
            f.close()

        for key in self.keys:
            logger.debug("{} shape {}".format(key, self.data[key].shape))

    def average_data(self):
        self.average = OrderedDict()
        self.std_dev = OrderedDict()
        for key in self.keys:
            self.average[key] = np.mean(self.data[key], axis=0)[0]
            self.std_dev[key] = np.std( self.data[key], axis=0)[0]

        for key in self.keys:
            logger.debug("{} shape {} and {}".format(key, self.average[key].shape, self.std_dev[key].shape))


class APJSingleColumnFigure():
    def __init__(self, aspect_ratio=None, lineplot=True, fontsize=8):
        import scipy.constants as scpconst
        import matplotlib.pyplot as plt

        self.plt = plt
        
        if aspect_ratio is None:
            self.aspect_ratio = scpconst.golden
        else:
            self.aspect_ratio = aspect_ratio

        if lineplot:
            self.dpi = 600
        else:
            self.dpi = 300
        
        self.fontsize=fontsize

        self.figure()
        self.add_subplot()
        self.set_fontsize(fontsize=fontsize)

    def figure(self):
            
        x_size = 3.5 # width of single column in inches
        y_size = x_size/self.aspect_ratio

        self.fig = self.plt.figure(figsize=(x_size, y_size))

    def add_subplot(self):
        self.ax = self.fig.add_subplot(1,1,1)

    def savefig(self, filename, dpi=None, **kwargs):
        if dpi is None:
            dpi = self.dpi

        self.plt.tight_layout(pad=0.25)
        self.fig.savefig(filename, dpi=dpi, **kwargs)

    def set_fontsize(self, fontsize=None):
        if fontsize is None:
            fontsize = self.fontsize

        for item in ([self.ax.title, self.ax.xaxis.label, self.ax.yaxis.label] +
             self.ax.get_xticklabels() + self.ax.get_yticklabels()):
            item.set_fontsize(fontsize)

    def legend(self, title=None, fontsize=None, **kwargs):
        if fontsize is None:
            self.legend_fontsize = apjfig.fontsize
        else:
            self.legend_fontsize = fontsize

        self.legend_object = self.ax.legend(prop={'size':self.legend_fontsize}, **kwargs)
        if title is not None:
            self.legend_object.set_title(title=title, prop={'size':self.legend_fontsize})

        return self.legend_object

def semilogy_posneg(ax, x, y, color=None,  color_pos=None, color_neg=None, **kwargs):
    pos_mask = np.logical_not(y>0)
    neg_mask = np.logical_not(y<0)
    pos_line = np.ma.MaskedArray(y, pos_mask)
    neg_line = np.ma.MaskedArray(y, neg_mask)

    if color is None:
        color = next(ax._get_lines.color_cycle)

    if color_pos is None:
        color_pos = color

    if color_neg is None:
        color_neg = color
        
    ax.semilogy(x, pos_line, color=color_pos, **kwargs)
    ax.semilogy(x, np.abs(neg_line), color=color_neg, linestyle='dashed')
