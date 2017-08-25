# README #

These are basic stratified fluid dynamics problems, for flows under
the fully compressible Navier-Stokes equations in a polytropic background, executed
with the [Dedalus](http://dedalus-project.org) pseudospectral
framework.  To run these problems, first install
[Dedalus](http://dedalus-project.org/) (and on
[bitbucket](https://bitbucket.org/dedalus-project/dedalus2)). 

## Initial Value Problems

Once [Dedalus](http://dedalus-project.org/) is installed and activated, do the following:
```
#!bash
mpirun -np 16 python3 FC_poly.py --run_time=0.1
mpirun -np 1 python3 plot_energies.py FC_poly_nrhocz3_Ra1e4_eps1e-4_a4/scalar/*.h5
mpirun -np 2 python3 plot_slices.py FC_poly_nrhocz3_Ra1e4_eps1e-4_a4/slices/*.h5
```

For multitropes, use
```
#!bash
mpirun -np 16 python3 FC_multi.py
```
and for exoplanet inspired (stable top, unstable bottom) atmospheres, consider
```
#!bash
mpirun -np 16 python3 FC_multi.py --oz
```

We are using docopt in these general driver cases, and
```
#!bash
python3 FC_multi.py --help
```
will describe various command-line options.

### Rotating Runs
To run a rotating polytrope, you could type, for example:
```
#!bash
mpirun -np 64 python3 FC_poly.py --3D --rotating --nz=32 --nx=32 --ny=32 --mesh=8,8
```
Which will make a 3D rotating atmosphere, where both the x- and z- directions are
broken up into 8 pieces (thus the 8,8 in the mesh).  

#### Pleiades unicode fix
The rotating equations contain some Unicode greek letters which are particularly 
unhappy in the pleiades environment.  In order to get around errors which
could be thrown by this, perform the following steps:

First, create a file in your home directory called 
*._my_mpi*.  Inside of that file, type
```
#!bash
#!/bin/bash
export LANG=en_US.UTF-8
/nasa/sgi/mpt/2.14r19/bin/mpiexec_mpt $*
```
Save and quit, then run 
```
#!bash
chmod +x $HOME/._my_mpi
```
to make that file executable.  Finally, add a line
to your .profile or .bashrc that aliases mpi to use this file:
```
#!bash
alias mpiexec_mpt="$HOME/._my_mpi"
```
Now, any time you call mpiexec_mpt, it will be properly wrapped in the right
language enviroment so that it understands unicode!  You could go through similar
steps to wrap python3, and that would allow you to use Unicode when running in serial, as well.

## Eigenvalue Problems (for finding onset of convection)

To find the critical Rayleigh number for convective onset in a low-Mach number
polytropic atmosphere, try
```
#!bash
mpirun -n 200 python3 FC_onset_curve.py --epsilon=1e-4
```

For a high-stiffness multitropic atmosphere, try
```
#!bash
mpirun -n 200 python3 FC_onset_curve.py --Multitrope --nz=96,48 --stiffness=1e3 --rayleigh_steps=20 --kx_steps=20 --kx_start=0.3 --kx_stop=10
```
Images of Rayleigh number / wavenumber space will be output, with the growth rate
of the maximum eigenvalue shown.

Contact the exoweather team for more details.

References: Brown, Vasil & Zweibel 2012, Vasil et al 2013, Lecoanet et al 2014.

