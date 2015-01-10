PROBLEMDIR=$(shell basename `dirname \`pwd\``)"/"$(shell basename `pwd`)
export OPT=-O3 -mmic -g
export LIB=-mmic
export OPENMP=1
export QUADRUPOLE=0
export CC=icc

all:
	# Setup link to different modules
	ln -fs gravity_phi.c ../../src/gravity.c
	ln -fs boundaries_open.c ../../src/boundaries.c
	ln -fs integrator_leapfrog.c ../../src/integrator.c
	ln -fs collisions_none.c ../../src/collisions.c
	# Setup link to problem file
	ln -fs ../$(PROBLEMDIR)/problem.c ../../src/problem.c
	# Compile
	$(MAKE) -C ../../src/
	# Copy result
	cp ../../src/nbody .

copy0:
	scp nbody mic0:~/
	scp /opt/intel/lib/mic/libiomp5.so mic0:~/

copy1:
	scp nbody mic1:~/
	scp /opt/intel/lib/mic/libiomp5.so mic1:~/
	# Now run
	# export LD_LIBRARY_PATH=~/

doc: all
	$(MAKE) -C ../../src/ doc

clean:
	$(MAKE) -C ../../src/ clean
	rm -vf nbody
