import numpy as np
import rebound
import copy

class State(object):

    def __init__(self, planets=None):
        self.logp = None
        self.logp_d = None
        self.logp_dd = None
        self.Nvars = 0
        self.Nplanets = 0
        if planets is not None:
            self.Nplanets = len(planets)
            for p in planets:
                self.Nvars += len(p)
            self.keys = []
            self.values = np.zeros(self.Nvars)
            vi = 0
            for i,planet in enumerate(planets):
                for k, v in planet.items():
                    self.keys.append("%d %s"%(i,k))
                    self.values[vi] = v
                    vi +=1


    def setup_sim(self):
        sim = rebound.Simulation()
        sim.ri_ias15.min_dt = 1e-1
        sim.add(m=1.)
        ds = [{}]*self.Nplanets
        for i in range(self.Nvars):
            kp, kv = self.keys[i].split()
            ds[int(kp)][kv] = self.values[i]
        for d in ds:
            sim.add(primary=sim.particles[0],**d)

        sim.move_to_com()
        return sim

    def get_rv(self, times):
        sim = self.setup_sim()
        
        rv = np.zeros(len(times))
        for i, t in enumerate(times):
            sim.integrate(t)
            rv[i] = sim.particles[0].vx

        return rv

    def get_rv_plotting(self, Npoints=200, tmax=1.5):
        times = np.linspace(0,tmax,Npoints)
        return times, self.get_rv(times)

    def get_chi2(self, obs):
        rv = self.get_rv(obs.t)
        chi2 = 0.
        for i, t in enumerate(obs.t):
            chi2 += (rv[i]-obs.rv[i])**2
        return chi2/(obs.error**2 * obs.Npoints)

    def get_logp(self, obs):
        if self.logp is None:
            self.logp = -self.get_chi2(obs)
        return self.logp
    
    def shift_params(self, vec):
        self.logp = None
        self.logp_d = None
        self.logp_dd = None
        if len(vec)!=self.Nvars:
            raise AttributeError("vector has wrong length")
        self.values += vec
   
    def get_rawkeys(self):
        return [k.split()[1] for k in self.keys]

    def deepcopy(self):
        s = State()
        s.Nvars = self.Nvars
        s.Nplanets = self.Nplanets
        s.keys = copy.deepcopy(self.keys)
        s.values = copy.deepcopy(self.values)
        return s

    def setup_sim_vars(self):
        sim = self.setup_sim()
        variations1 = []
        variations2 = []
        for vindex in range(self.Nvars):
            pindex, vname = self.keys[vindex].split()
            pindex = int(pindex)+1
            v = sim.add_variation(order=1)
            v.vary_pal(pindex,vname)
            variations1.append(v)
        
        for vindex1 in range(self.Nvars):
            for vindex2 in range(self.Nvars):
                if vindex1 >= vindex2:
                    pindex1, vname1 = self.keys[vindex1].split()
                    pindex1 = int(pindex1)+1
                    pindex2, vname2 = self.keys[vindex2].split()
                    pindex2 = int(pindex2)+1
                    v = sim.add_variation(order=2, first_order=variations1[vindex1], first_order_2=variations1[vindex2])
                    if pindex1 == pindex2:
                        v.vary_pal(pindex1,vname1,vname2)
                    variations2.append(v)

        sim.move_to_com()
        return sim, variations1, variations2

    def get_chi2_d_dd(self, obs):
        sim, variations1, variations2 = self.setup_sim_vars()
        chi2 = 0.
        chi2_d = np.zeros(self.Nvars)
        chi2_dd = np.zeros((self.Nvars,self.Nvars))
        normfac = 1./(obs.error**2 * obs.Npoints)
        for i, t in enumerate(obs.t):
            sim.integrate(t)
            chi2 += (sim.particles[0].vx-obs.rv[i])**2*normfac
            v2index = 0
            for vindex1 in range(self.Nvars):
                chi2_d[vindex1] += 2. * variations1[vindex1].particles[0].vx * (sim.particles[0].vx-obs.rv[i])*normfac
            
                for vindex2 in range(self.Nvars):
                    if vindex1 >= vindex2:
                        chi2_dd[vindex1][vindex2] +=  2. * variations2[v2index].particles[0].vx * (sim.particles[0].vx-obs.rv[i])*normfac + 2. * variations1[vindex1].particles[0].vx * variations1[vindex2].particles[0].vx*normfac
                        v2index += 1
                        chi2_dd[vindex2][vindex1] = chi2_dd[vindex1][vindex2]

        return chi2, chi2_d, chi2_dd
    
    def get_logp_d_dd(self, obs):
        if (self.logp is None) or (self.logp_d is None) or (self.logp_dd is None):
            chi, chi_d, chi_dd = self.get_chi2_d_dd(obs)
            self.logp, self.logp_d, self.logp_dd = -chi, -chi_d, -chi_dd
        return self.logp, self.logp_d, self.logp_dd



