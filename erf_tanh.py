import numpy as np
import scipy.special as scp
import matplotlib.pyplot as plt

import dedalus.public as de

def phi(x, f, center=0, width=0.02):
    return (1/2*(1-f((x-center)/width)))

plt.style.use('ggplot')
nx = 1024
Lx = 2
x_basis = de.Chebyshev('x', nx, interval=[0., Lx], dealias=3/2)

domain = de.Domain([x_basis])
x = domain.grid(0)

phi_tanh = domain.new_field()
phi_tanh['g'] = phi(x, np.tanh, center=1)

phi_erf = domain.new_field()
phi_erf['g'] = phi(x, scp.erf, center=1)


fig = plt.figure()
ax = fig.add_subplot(2,1,1)
ax.plot(x, phi_erf['g'],  label='f(x)=erf(x)')
ax.plot(x, phi_tanh['g'], label='f(x)=tanh(x)')
ax.legend(loc='upper right', title=r'$\phi(x) = \frac{1}{2}(1-f((x-1)/0.02))$')
ax.set_ylim(-0.05, 1.05)

ax = fig.add_subplot(2,1,2)
ax.loglog(np.abs(phi_erf['c']), label='f(x)=erf(x)', marker='o', linestyle='none')
ax.loglog(np.abs(phi_tanh['c']), label='f(x)=tanh(x)', marker='.', linestyle='none')
ax.set_xlim(1, 1024)

fig.savefig('erf_tanh.png', dpi=600)

