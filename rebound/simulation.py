from ctypes import *
from . import clibrebound
from .particle import *
from .units import units_convert_particle, check_units, convert_G
import math
import os
import ctypes.util
try:
    import pkg_resources
except: 
    # Fails on python3, but not important
    pass
import types
      
### The following enum and class definitions need to
### consitent with those in rebound.h
        
INTEGRATORS = {"ias15": 0, "whfast": 1, "sei": 2, "wh": 3, "leapfrog": 4, "hybrid": 5, "none": 6}

class reb_vec3d(Structure):
    _fields_ = [("x", c_double),
                ("y", c_double),
                ("z", c_double)]

class reb_dp7(Structure):
    _fields_ = [("p0", POINTER(c_double)),
                ("p1", POINTER(c_double)),
                ("p2", POINTER(c_double)),
                ("p3", POINTER(c_double)),
                ("p4", POINTER(c_double)),
                ("p5", POINTER(c_double)),
                ("p6", POINTER(c_double))]

class reb_ghostbox(Structure):
    _fields_ = [("shiftx", c_double),
                ("shifty", c_double),
                ("shiftz", c_double),
                ("shiftvx", c_double),
                ("shiftvy", c_double),
                ("shiftvz", c_double)]


class reb_simulation_integrator_hybrid(Structure):
    _fields_ = [("switch_ratio", c_double),
                ("mode", c_int)]

class reb_simulation_integrator_wh(Structure):
    _fields_ = [(("allocatedN"), c_int),
                ("eta", POINTER(c_double))]

class reb_simulation_integrator_sei(Structure):
    _fields_ = [("OMEGA", c_double),
                ("OMEGAZ", c_double),
                ("lastdt", c_double),
                ("sindt", c_double),
                ("tandt", c_double),
                ("sindtz", c_double),
                ("tandtz", c_double)]

class reb_simulation(Structure):
    pass
reb_simulation._fields_ = [("t", c_double),
                ("G", c_double),
                ("softening", c_double),
                ("dt", c_double),
                ("dt_last_done", c_double),
                ("N", c_int),
                ("N_var", c_int),
                ("N_active", c_int),
                ("allocated_N", c_int),
                ("exit_simulation", c_int),
                ("exact_finish_time", c_int),
                ("force_is_velocity_dependent", c_uint),
                ("gravity_ignore_10", c_uint),
                ("output_timing_last", c_double),
                ("boxsize", reb_vec3d),
                ("boxsize_max", c_double),
                ("root_size", c_double),
                ("root_n", c_int),
                ("root_nx", c_int),
                ("root_ny", c_int),
                ("root_nz", c_int),
                ("nghostx", c_int),
                ("nghosty", c_int),
                ("nghostz", c_int),
                ("collisions", c_void_p),
                ("collisions_allocatedN", c_int),
                ("minimum_collision_celocity", c_double),
                ("collisions_plog", c_double),
                ("max_radius", c_double*2),
                ("collisions_Nlog", c_long),
                ("calculate_megno", c_int),
                ("megno_Ys", c_double),
                ("megno_Yss", c_double),
                ("megno_cov_Yt", c_double),
                ("megno_var_t", c_double),
                ("megno_mean_t", c_double),
                ("megno_mean_Y", c_double),
                ("megno_n", c_long),
                ("collision", c_int),
                ("integrator", c_int),
                ("boundary", c_int),
                ("gravity", c_int),
                ("particles", POINTER(Particle)),
                ("gravity_cs", POINTER(reb_vec3d)),
                ("gravity_cs_allocatedN", c_int),
                ("tree_root", c_void_p),
                ("opening_angle2", c_double),
                ("ri_whfast", c_int),  #// TODO!!
                ("ri_ias15", c_int), #// TODO!!
                ("ri_sei", reb_simulation_integrator_sei), 
                ("ri_wh", reb_simulation_integrator_wh), 
                ("ri_hybrid", reb_simulation_integrator_hybrid),
                ("additional_forces", CFUNCTYPE(None,POINTER(reb_simulation))),
                ("post_timestep_modifications", CFUNCTYPE(None,POINTER(reb_simulation))),
                ("heartbeat", CFUNCTYPE(None,POINTER(reb_simulation))),
                ("coefficient_of_restitution", CFUNCTYPE(c_double,POINTER(reb_simulation), c_double)),
                ("collisions_resolve", CFUNCTYPE(None,POINTER(reb_simulation), c_void_p)),
                 ]


class Simulation(object):
    simulation = None
    def __init__(self):
        clibrebound.reb_create_simulation.restype = POINTER(reb_simulation)
        self.simulation = clibrebound.reb_create_simulation()
    
    AFF = CFUNCTYPE(None)
    afp = None # additional forces pointer
    ptmp = None # post timestep modifications pointer 
    _units = {'length':None, 'time':None, 'mass':None}

# Status functions
    def status(self):
        """ Returns a string with a summary of the current status 
            of the simulation
            """
        s= ""
        s += "---------------------------------\n"
        try:
            s += "Rebound version:     \t" + pkg_resources.require("rebound")[0].version +"\n"
        except:
            # Fails on python3, but not important
            pass
        s += "Number of particles: \t%d\n" %self.N       
        s += "Selected integrator: \t" + self.integrator + "\n"       
        s += "Simulation time:     \t%f\n" %self.t
        s += "Current timestep:    \t%f\n" %self.dt
        if self.N>0:
            s += "---------------------------------\n"
            for p in self.particles:
                s += str(p) + "\n"
        s += "---------------------------------"
        print(s)

# Set function pointer for additional forces
    @property
    def additional_forces(self):
        return self.afp   # might not be needed

    @additional_forces.setter
    def additional_forces(self, func):
        if(isinstance(func,types.FunctionType)):
            # Python function pointer
            self.afp = self.AFF(func)
            self.clibrebound.set_additional_forces(self.afp)
        else:
            # C function pointer
            self.clibrebound.set_additional_forces_with_parameters(func)
            self.afp = "C function pointer value currently not accessible from python.  Edit librebound.py"

    @property
    def post_timestep_modifications(self):
        return self.ptmp

    @post_timestep_modifications.setter
    def post_timestep_modifications(self, func):
        if(isinstance(func, types.FunctionType)):
            # Python function pointer
            self.ptmp = self.AFF(func)
            self.clibrebound.set_post_timestep_modifications(self.ptmp)
        else:
            # C function pointer
            self.clibrebound.set_post_timestep_modifications_with_parameters(func)
            self.ptmp = "C function pointer value currently not accessible from python.  Edit librebound.py" 

# Setter/getter of parameters and constants
    @property 
    def dt(self):
        return self.simulation.contents.dt

    @dt.setter
    def dt(self, value):
        self.simulation.contents.dt = c_double(value)

    @property 
    def t(self):
        return self.simulation.contents.t
    @t.setter
    def t(self, value):
        self.simulation.contents.t = c_double(value)
    
    @property 
    def N_active(self):
        return self.simulation.contents.N_active
    @N_active.setter
    def N_active(self, value):
        self.simulation.contents.N_active = c_int(value)

    @property 
    def N(self):
        return self.simulation.contents.N

    @property
    def integrator(self):
        i = self.simulation.contents.integrator
        for name, _i in INTEGRATORS.items():
            if i==_i:
                return name
        return i

    @integrator.setter
    def integrator(self, value):
        if isinstance(value, int):
            self.simulation.contents.integrator = c_int(value)
        elif isinstance(value, basestring):
            debug.integrator_fullname = value
            debug.integrator_package = "REBOUND"
            value = value.lower()
            if value in INTEGRATORS: 
                self.integrator = INTEGRATORS[value]
            elif value.lower() == "mercury":
                debug.integrator_package = "MERCURY"
            elif value.lower() == "swifter-whm":
                debug.integrator_package = "SWIFTER"
            elif value.lower() == "swifter-symba":
                debug.integrator_package = "SWIFTER"
            elif value.lower() == "swifter-helio":
                debug.integrator_package = "SWIFTER"
            elif value.lower() == "swifter-tu4":
                debug.integrator_package = "SWIFTER"
            else:
                raise ValueError("Warning. Integrator not found.")

    @property
    def force_is_velocitydependent(self):
        return c_int.in_dll(self.clibrebound, "integrator_force_is_velocitydependent").value

    @force_is_velocitydependent.setter
    def force_is_velocitydependent(self, value):
        if isinstance(value, int):
            c_int.in_dll(self.clibrebound, "integrator_force_is_velocitydependent").value = value
            return
        raise ValueError("Expecting integer.")
    
# Units

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, newunits):
        newunits = check_units(newunits)        
        if self.particles: # some particles are loaded
            raise Exception("Error:  You cannot set the units after populating the particles array.  See Units.ipynb in python_tutorials.")
        self.update_units(newunits) 


    def update_units(self, newunits): 
        self._units['length'] = newunits[0] 
        self._units['time'] = newunits[1] 
        self._units['mass'] = newunits[2] 
        self.G = convert_G(self._units['length'], self._units['time'], self._units['mass'])

    def convert_particle_units(self, *args): 
        new_l, new_t, new_m = check_units(args)
        for p in self.particles:
            self.convert_p(p, self._units['length'], self._units['time'], self._units['mass'], new_l, new_t, new_m)
        self.update_units((new_l, new_t, new_m))

# MEGNO
    def init_megno(self, delta):
        self.clibrebound.tools_megno_init(c_double(delta))
    
    def calculate_megno(self):
        self.clibrebound.tools_megno.restype = c_double
        return self.clibrebound.tools_megno()
    
    def calculate_lyapunov(self):
        self.clibrebound.tools_lyapunov.restype = c_double
        return self.clibrebound.tools_lyapunov()
    
    @property
    def N_megno(self):
        return c_int.in_dll(self.clibrebound,"N_megno").value 
    
# Particle add function, used to be called particle_add() and add_particle() 
    def add(self, particle=None, **kwargs):   
        """Adds a particle to REBOUND. Accepts one of the following:
        1) A single Particle structure.
        2) The particle's mass and a set of cartesian coordinates: m,x,y,z,vx,vy,vz.
        3) The primary as a Particle structure, the particle's mass and a set of orbital elements primary,a,anom,e,omega,inv,Omega,MEAN (see kepler_particle() for the definition of orbital elements). 
        4) A name of an object (uses NASA Horizons to look up coordinates)
        5) A list of particles or names.
        """
        if particle is not None:
            if isinstance(particle, Particle):
                if kwargs == {}: # copy particle
                    clibrebound.reb_add(self.simulation, particle)
                else: # use particle as primary
                    self.add(Particle(simulation=self.simulation, primary=particle, **kwargs))
            elif isinstance(particle, list):
                for p in particle:
                    self.add(p)
            elif isinstance(particle,str):
                if None in self.units.values():
                    self.units = ('AU', 'yr2pi', 'Msun')
                self.add(horizons.getParticle(particle,**kwargs))
                units_convert_particle(self.particles[-1], 'km', 's', 'kg', self._units['length'], self._units['time'], self._units['mass'])
        else: 
            self.add(Particle(simulation=self.simulation, **kwargs))

# Particle getter functions
    @property
    def particles(self):
        """Return an array that points to the particle structure.
        This is an array of pointers and thus the contents of the array update 
        as the simulation progresses. Note that the pointers could change,
        for example when a particle is added or removed from the simulation. 
        """
        ps = []
        N = self.N 
        ps_a = self.simulation.contents.particles
        for i in range(0,N):
            ps.append(ps_a[i])
        return ps

    @particles.deleter
    def particles(self):
        self.clibrebound.particles_remove_all()

    def remove_particle(self, index=None, ID=None, keepSorted=1):
        """ Removes a particle from the simulation.

        Parameters

        ----------

        Either the index in the particles array to remove, or the ID of the particle to
        remove.  The keepSorted flag ensures the particles array remains sorted
        in order of increasing IDs.  One might set it to zero in cases with many
        particles and many removals to speed things up.
        """
        if index is not None:
            success = self.clibrebound.particles_remove(c_int(index), keepSorted)
            if not success:
                print("Index %d passed to remove_particle was out of range (N=%d). Did not remove particle.\n"%(index, self.N))
            return
        if ID is not None:
            success = self.clibrebound.particles_remove_ID(c_int(ID), keepSorted)
            if not success:
                print("ID %d passed to remove_particle was not found.  Did not remove particle.\n"%(ID))


# Orbit calculation
    def calculate_orbits(self, heliocentric=False):
        """ Returns an array of Orbits of length N-1.

        Parameters
        __________

        By default this functions returns the orbits in Jacobi coordinates. 
        Set the parameter heliocentric to True to return orbits in heliocentric coordinates.
        """
        _particles = self.particles
        orbits = []
        for i in range(1,self.N):
            if heliocentric:
                com = _particles[0]
            else:
                com = self.calculate_com(i)
            orbits.append(_particles[i].calculate_orbit(self.simulation, primary=com))
        return orbits

# COM calculation 
    def calculate_com(self, last=None):
        """Returns the center of momentum for all particles in the simulation"""
        m = 0.
        x = 0.
        y = 0.
        z = 0.
        vx = 0.
        vy = 0.
        vz = 0.
        ps = self.particles    # particle pointer
        if last is not None:
            last = min(last, self.N)
        else:
            last = self.N
        for i in range(last):
            m  += ps[i].m
            x  += ps[i].x*ps[i].m
            y  += ps[i].y*ps[i].m
            z  += ps[i].z*ps[i].m
            vx += ps[i].vx*ps[i].m
            vy += ps[i].vy*ps[i].m
            vz += ps[i].vz*ps[i].m
        if m>0.:
            x /= m
            y /= m
            z /= m
            vx /= m
            vy /= m
            vz /= m
        return Particle(m=m, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
    

# Tools
    def move_to_com(self):
        clibrebound.reb_move_to_com(self.simulation)
    
    def calculate_energy(self):
        self.clibrebound.tools_energy.restype = c_double
        return self.clibrebound.tools_energy()

# Input/Output routines
    def save(self, filename):
        self.clibrebound.output_binary(c_char_p(filename.encode("ascii")))
        
    def load(self, filename):
        self.clibrebound.input_binary(c_char_p(filename.encode("ascii")))
        
# Integrator Flags
    @property 
    def integrator_whfast_corrector(self):
        return c_int.in_dll(self.clibrebound, "integrator_whfast_corrector").value
    
    @integrator_whfast_corrector.setter 
    def integrator_whfast_corrector(self, value):
        c_int.in_dll(self.clibrebound, "integrator_whfast_corrector").value = value

    @property
    def integrator_whfast_safe_mode(self):
        return c_int.in_dll(self.clibrebound, "integrator_whfast_safe_mode").value

    @integrator_whfast_safe_mode.setter
    def integrator_whfast_safe_mode(self, value):
        c_int.in_dll(self.clibrebound, "integrator_whfast_safe_mode").value = value

    @property
    def recalculate_jacobi_this_timestep(self):
        return c_int.in_dll(self.clibrebound, "integrator_whfast_recalculate_jacobi_this_timestep").value

    @recalculate_jacobi_this_timestep.setter
    def recalculate_jacobi_this_timestep(self, value):
        c_int.in_dll(self.clibrebound, "integrator_whfast_recalculate_jacobi_this_timestep").value = value
    
# Integration

    def step(self, do_timing = 1):
        self.clibrebound.rebound_step(c_int(do_timing))

    def integrate(self, tmax, exact_finish_time=1, maxR=0., minD=0.):
        # TODO: Fix minD maxR
        if debug.integrator_package =="REBOUND":
            clibrebound.reb_integrate.restype = c_int
            ret_value = clibrebound.reb_integrate(self.simulation, c_double(tmax))
            if ret_value == 1:
                raise self.NoParticleLeft("No more particles left in simulation.")
            if ret_value == 2:
                raise self.ParticleEscaping(c_int.in_dll(self.clibrebound, "escapedParticle").value, self.t)
            if ret_value == 3:
                raise self.CloseEncounter(c_int.in_dll(self.clibrebound, "closeEncounterPi").value,
                                     c_int.in_dll(self.clibrebound, "closeEncounterPj").value, self.t)
        else:
            debug.integrate_other_package(tmax,exact_finish_time)

    def integrator_synchronize(self):
        self.clibrebound.integrator_synchronize()

# Exceptions    
    class CloseEncounter(Exception):
        def __init__(self, index1, index2, t):
                self.index1 = index1
                self.index2 = index2
                self.t = t
        def __str__(self):
                return "A close encounter occured at time %f between particles with indices %d and %d."%(self.t, self.index1,self.index2)

    class ParticleEscaping(Exception):
        def __init__(self, index, t):
            self.index = index
            self.t = t
        def __str__(self):
            return "At least one particle's distance > maxR at time %f (index = %d)."%(self.t, self.index)

    class NoParticleLeft(Exception):
        pass
    


# Import at the end to avoid circular dependence
from . import horizons
from . import debug
