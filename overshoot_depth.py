import matplotlib.pyplot as plt
import numpy as np

def L_ov(stiffness, n_rho_cz):
    m_rz = 3
    m_ad = 1.5
    epsilon = (3-m_ad)/stiffness    
    m_cz = m_ad - epsilon
    Lz_cz = np.exp(n_rho_cz/m_cz)-1
    T_bcz = Lz_cz+1
    L_ov = ((T_bcz)**((stiffness+1)/stiffness) - T_bcz)*(m_rz+1)/(m_cz+1)
    return L_ov

stiffness = np.logspace(1,5)
n_rho_set = [1,3.5,5]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for n_rho in n_rho_set:
    ax.loglog(stiffness, L_ov(stiffness, n_rho), label=r"$n_\rho$={}".format(n_rho))
ax.legend()
    
fig.savefig("overshoot_depth.png", dpi=600)
