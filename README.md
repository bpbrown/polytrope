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
python3 anelastic_igw.py
python3 plot_results_parallel.py anelastic_igw slices 1 1 5
```
To run in parallel, do something like
```
#!bash
mpirun -np 4 python3 anelastic_convection.py
mpirun -np 2 python3 plot_results_parallel.py anelastic_convection slices 1 1 10
```

Contact Ben Brown (and see Brown et al 2012; Vasil et al 2013) for
more details.

