# In this example, we initialize a sparse circumplanetary disk of satellites 
# around a planet orbiting the Sun. The initial outer edge of the disk extends 
# to the Hill radius of the planet. We integrate to study the evolution of 
# satellites under the influence of the planet and solar gravitational potentials.
#
# Example by Ryan Cloutier.

import rebound
import matplotlib; matplotlib.use("pdf")
import matplotlib.pyplot as plt   
import numpy as np
from compute_satellite_orbits import compute_satellite_orbits as cso
import os

# Function to draw random orbital elements within fixed intervals
def get_test_orbit(rHill):
    # semimajor axis is between .1 and 2 Hill radii
    a     = rHill * (np.random.rand()*(1-.1) + .1)
    ecc   = np.random.rand()*.1
    inc   = np.deg2rad(np.random.rand()*10.)
    Omega = np.random.rand()*2*np.pi
    omega = np.random.rand()*2*np.pi
    f     = np.random.rand()*2*np.pi
    return a, ecc, inc, Omega, omega, f

# Create a simulation with 'nice' units
sim = rebound.Simulation()
sim.units = ["AU", "MSun", "day"]

# Use IAS15 for the adaptive time-stepping
sim.integrator = 'ias15'

# Add the Sun and Jupiter on a circular orbit
Sun = rebound.Particle(m=1.)
sim.add(Sun)
sim.add(primary=Sun, m=1e-3, a=5., id=1) 
Jup = sim.particles[1]

# Add satellites with random eccentricities, inclinations, and 
# joviancentric semimajor axes the between .1 and 2 Hill radii
rHill = Jup.a * (Jup.m/(3.*Sun.m))**(1./3) 
Ntest = 100
i = 0
while i < Ntest:
    # Draw random orbital elements
    testa,testecc,testinc,testOmega,testomega,testf=get_test_orbit(rHill)
    sim.add(primary=Jup, m=0., a=testa, e=testecc, inc=testinc,
            Omega=testOmega, omega=testomega, f=testf, id=i+2)
    i += 1

# Move to the system's centre-of-mass
sim.move_to_com()

# Integrate for 1 year and save outputs 
os.system('mkdir -p output')
Noutput = 100
tmax = 1  # 1 year
orbital_parameters = np.zeros((Noutput,Ntest,3))
particles = sim.particles
for i in range(Noutput):
    tfin = 365.25 * tmax/Noutput * i
    print 'Integrating to t = %.3e days'%tfin
    sim.integrate(tfin)

    # Save test particle orbits
    for j in range(Ntest):
        orbital_parameters[i][j][0] = sim.t 
        orbital_parameters[i][j][1] = particles[j+2].calculate_orbit(primary=particles[1]).a
        orbital_parameters[i][j][2] = particles[j+2].e

# Plot the evolution of a random subset of satellites
Nplot = 6
indices = np.arange(Ntest)
np.random.shuffle(indices)

plt.figure('evolution')
plt.subplot(211)
plt.plot(orbital_parameters[:,:,0], orbital_parameters[:,:,1]/rHill, '-', lw=2)
plt.plot([0,tfin], np.ones(2), 'k--')
plt.xlim((0,tfin))
plt.ylim((0,2))
plt.ylabel('Semimajor Axis\n(Jupiter Hill radii)')
plt.title('Satellite orbital evolution')
plt.subplot(212)
plt.plot(orbital_parameters[:,:,0], orbital_parameters[:,:,2], '-', lw=2)
plt.xlim((0,tfin))
plt.xlabel('Time (days)')
plt.ylabel('Eccentricity')
plt.savefig('orbit_evolution.png')
plt.close('evolution')

# Open the plot (Linux only)
from sys import platform
if platform == 'linux2':
    os.system('eog orbit_evolution.png')
elif platform == 'darwin':
    os.system('open orbit_evolution.png')

