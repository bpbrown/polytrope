import numpy as np
import scipy.special as scp
import matplotlib.pyplot as plt

import dedalus.public as de

def phi(x, f=scp.erf, center=0, width=0.02):
    return (1/2*(1-f((x-center)/width)))

plt.style.use('ggplot')

def erf_tanh_cheby(nx, Lx, center, width, fig_name='erf_tanh_cheby.png'):
    x_basis = de.Chebyshev('x', nx, interval=[0., Lx], dealias=3/2)

    domain = de.Domain([x_basis])

    make_plots(domain, fig_name=fig_name)
    
def erf_tanh_compound(nx, intervals, center, width, fig_name='erf_tanh_compound.png'):

    x_basis_list = []
    for i in range(len(nx)):
        print('sub interval {} : {} (nx={})'.format(i, intervals[i], nx[i]))
        x_basis = de.Chebyshev('x', nx[i], interval=intervals[i], dealias=3/2)
        x_basis_list.append(x_basis)

    x_basis = de.Compound('x', tuple(x_basis_list),  dealias=3/2)
    domain = de.Domain([x_basis])

    make_plots(domain, fig_name=fig_name)
 
def make_plots(domain, fig_name = 'erf_tanh.png'):
    x = domain.grid(0)
    
    phi_tanh = domain.new_field()
    phi_tanh['g'] = phi(x, f=np.tanh, center=center, width=width)

    phi_erf = domain.new_field()
    phi_erf['g'] = phi(x, f=scp.erf, center=center, width=width)

    fig2 = plt.figure()
    ax1 = fig2.add_subplot(2,1,1)
    ax1.plot(x, phi_erf['g'],  label='f(x)=erf(x)')
    ax1.plot(x, phi_tanh['g'], label='f(x)=tanh(x)')
    ax1.legend(loc='upper right', title=r'$\phi(x) = \frac{1}{2}(1-f((x-'+'{})/{}'.format(center, width)+r'))$')
    ax1.set_ylim(-0.05, 1.05)

    ax2 = fig2.add_subplot(2,1,2)
    ax2.loglog(np.abs(phi_erf['c'])/np.max(np.abs(phi_erf['c'])), label='f(x)=erf(x)', marker='o', linestyle='none')
    ax2.loglog(np.abs(phi_tanh['c'])/np.max(np.abs(phi_tanh['c'])), label='f(x)=tanh(x)', marker='.', linestyle='none')
    ax2.set_xlim(1, 1024)


    fig2.savefig(fig_name, dpi=600)

if __name__=="__main__":
    
    nx = 1024
    Lx = 2
    center = 0.75
    width=0.02
    print("single domain test")
    erf_tanh_cheby(nx, Lx, center, width)

    print("compound domain test")
    pad = 5
    nx = [384, 128, 128, 384]
    intervals = [[0., center-pad*width], [center-pad*width, center], [center, center+pad*width], [center+pad*width, Lx]]
    
    erf_tanh_compound(nx, intervals, center, width)


    print("compound domain test")
    pad = 5
    nx = [384,256, 384]
    intervals = [[0., center-pad*width], [center-pad*width, center+pad*width], [center+pad*width, Lx]]
    
    erf_tanh_compound(nx, intervals, center, width, fig_name='erf_tanh_centered_compound.png')

    
    print("simple compound domain test")
    nx = [512, 512]
    intervals = [[0., center], [center, Lx]]
    
    erf_tanh_compound(nx, intervals, center, width, fig_name='erf_tanh_simple_compound.png')
