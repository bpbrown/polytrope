# README #

These are basic stratified fluid dynamics problems, for flows under
the LBR anelastic approximation in a polytropic background, executed
with the [Dedalus](http://dedalus-project.org) pseudospectral
framework.  To run these problems, first install
[Dedalus](http://dedalus-project.org/) (and on
[bitbucket](https://bitbucket.org/dedalus-project/dedalus2)). 

Once [Dedalus](http://dedalus-project.org/) is installed and activated, do the following:
```
#!bash
mpirun -np 4 python3 FC_poly.py
mpirun -np 2 python3 plot_results_parallel.py FC_poly_Ra1e6 slices 1 1 10
```

For multitropes, use
```
#!bash
mpirun -np 4 python3 FC_multi.py
```
and for exoplanet inspired (stable top, unstable bottom) atmospheres, consider
```
#!bash
mpirun -np 4 python3 FC_poly_oz.py
```

We are using docopt in these general driver cases, and
```
#!bash
python3 FC_multi.py --help
```
will describe various command-line options.

Contact the exoweather team for more details.

References: Brown, Vasil & Zweibel 2012, Vasil et al 2013, Lecoanet et al 2014.

