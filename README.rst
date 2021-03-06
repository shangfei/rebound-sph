REBOUND-SPH - A multi-purpose SPH / N-body hybrid code
======================================================

IMPORTANT NOTICE
----------------

This repository is undergoing active developement. Currently, there are three examples starting with "sph_" under the examples directory. Type make in your terminal to give it a try. Note you may need to modify the Makefile to get it work. And the protoplanetary disk example is not included yet. If you have troubles to get started, feel free to write me an email (if you know me) or simply create an issue using the "Issue" tab.


INTRODUCTION
------------

REBOUND-SPH is a smoothed particle hydrodynamics (SPH) / N-body hybrid code. It inherits the modular structure and functions from the N-body integrator REBOUND, which is mainly developed by Hanno Rein as well as several major contributors including myself. With the newly integrated SPH scheme, one can use REBOUND-SPH to study fluid dynamics from the Lagrangian perspective, i.e. properties of fluid (such as density and pressure) are tracked following each moving fluid parcel (an SPH particle). An SPH particle is fundamentally different from an N-body particle: 1) an N-body particle represents a discrete bundle of mass and interacts with other particles through gravity; 2) an SPH particle represents a spatial distribution of mass using the "smoothing kernel" (similar to a Gaussian function) and as a matter of fact, the physical density at a given point is the summation of all overlapping contribution from nearby SPH particles; 3) SPH particles interacts with each via pressure as well as gravity (optional). In REBOUND-SPH, user has the freedom to run pure SPH simulations or SPH / N-body hybrid simulations. For pure N-body simulations, user is recommended to use REBOUND.

* The code is written entirely in C, conforms to the ISO standard C99 and can be used as a thread-safe shared library
* Real-time, 3D OpenGL visualization (C version)
* Parallelized with OpenMP (for shared memory systems)
* Parallelized with MPI using an essential tree for gravity and collisions (for distributed memory systems)
* No configuration is needed to run any of the example problems. Just type `make && ./rebound` in the problem directory to run them
* Comes with standard ASCII, REBOUND binary and HDF5 output routines 

EXAMPLES
--------
A Jupiter-mass planet embedded in a 0.02 solar mass protoplanetary disk. The simulation uses 2 million SPH particles, and the image is rendered with SPLASH.

.. image:: images/SPLASH_REBOUND-SPH.png
    :align: center

A Jupiter mass planet (the moving object) encounters a Solar mass star (the fixed point) at a distance about one Solar radius and gets tidally disrupted. This is a very low resolution simulation done on my laptop. The planet is simulated with only 2,500 SPH particles. However, the behavior of the tidal stream is well captured. 

.. image:: images/tidal_disruption.gif
    :align: center

And this is an AMR simulation done by myself before (`Liu et al. 2013 <http://adsabs.harvard.edu/abs/2013ApJ...762...37L>`_). Look how similar the accreted streams are. Note that the grid simulation was done on a cluster with more than a hundred CPUs.

.. image:: images/tidal_disruption_grid.png
    :align: center
    :scale: 50 %