import matplotlib.pyplot as plt

class APJSingleColumnFigure():
    def __init__(self, aspect_ratio=None, lineplot=True, fontsize=8):
        import scipy.constants as scpconst
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

        self.fig = plt.figure(figsize=(x_size, y_size))

    def add_subplot(self):
        self.ax = self.fig.add_subplot(1,1,1)

    def savefig(self, filename, dpi=None, **kwargs):
        if dpi is None:
            dpi = self.dpi

        plt.tight_layout(pad=0.25)
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

def semilogy_posneg(ax, x, y, color=None, **kwargs):
    pos_mask = np.logical_not(y>0)
    neg_mask = np.logical_not(y<0)
    pos_line = np.ma.MaskedArray(y, pos_mask)
    neg_line = np.ma.MaskedArray(y, neg_mask)

    if color is None:
        color = next(ax._get_lines.color_cycle)

    ax.semilogy(x, pos_line, color=color, **kwargs)
    ax.semilogy(x, np.abs(neg_line), color=color, linestyle='dashed')
