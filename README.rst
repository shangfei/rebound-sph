REBOUND-SPH - A multi-purpose SPH / N-body hybrid code
==================================================

INTRODUCTION
--------

REBOUND-SPH is a smoothed particle hydrodynamics (SPH) / N-body hybrid code. It inherits the modular structure and functions from the N-body integrator REBOUND, which is mainly developed by Hanno Rein as well as several major contributors including myself. With the newly integrated SPH scheme, one can use REBOUND-SPH to study fluid dynamics from the Lagrangian perspective, i.e. properties of fluid (such as density and pressure) are tracked following each moving fluid parcel (an SPH particle). An SPH particle is fundamentally different from an N-body particle: 1) an N-body particle represents a discrete bundle of mass and interacts with other particles through gravity; 2) an SPH particle represents a spatial distribution of mass using the "smoothing kernel" (similar to a Gaussian function) and as a matter of fact, the physical density at a given point is the summation of all overlapping contribution from nearby SPH particles; 3) SPH particles interacts with each via pressure as well as gravity (optional). In REBOUND-SPH, user has the freedom to run pure SPH simulations or SPH / N-body hybrid simulations. For pure N-body simulations, user is recommended to use REBOUND.

* The code is written entirely in C, conforms to the ISO standard C99 and can be used as a thread-safe shared library
* Real-time, 3D OpenGL visualization (C version)
* Parallelized with OpenMP (for shared memory systems)
* Parallelized with MPI using an essential tree for gravity and collisions (for distributed memory systems)
* No configuration is needed to run any of the example problems. Just type `make && ./rebound` in the problem directory to run them
* Comes with standard ASCII, REBOUND binary and HDF5 output routines 
