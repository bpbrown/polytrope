# README #

These are basic stratified fluid dynamics problems, for flows under
the fully compressible Navier-Stokes equations in a polytropic background, executed
with the [Dedalus](http://dedalus-project.org) pseudospectral
framework.  To run these problems, first install
[Dedalus](http://dedalus-project.org/) (and on
[bitbucket](https://bitbucket.org/dedalus-project/dedalus2)). 

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

Contact the exoweather team for more details.

References: Brown, Vasil & Zweibel 2012, Vasil et al 2013, Lecoanet et al 2014.

