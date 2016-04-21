import numpy as np
import emcee
from scipy import stats

class Mcmc(object):
    def __init__(self, initial_state, obs):
        self.state = initial_state.deepcopy()
        self.obs = obs

    def step(self):
        return True 
    
    def step_force(self):
        tries = 1
        while self.step()==False:
            tries += 1
            pass
        return tries

def lnprob(x, e):
    e.state.set_params(x)
    logp = e.state.get_logp(e.obs)
    return logp


def update_one(param):
    self, s1, sg2 = param
    #s1 = sg2[0].deepcopy()
    #s1.set_params(s1p)
    rejections = 0
    while True:
        s2 = np.random.choice(sg2)
        proposal = self.generate_proposal(s1, s2)
        logp = s1.get_logp(self.obs)
        logp_proposal = proposal.get_logp(self.obs)
        if logp_proposal-logp>np.log(np.random.uniform()):
            s1.set_params(proposal.get_params())
            return rejections 
        else:
            rejections += 1

class PalEns(Mcmc):
    def __init__(self, initial_state, obs, scales, nwalkers=10):
        super(PalEns,self).__init__(initial_state, obs)
        self.set_scales(scales)
        self.nwalkers = nwalkers
        self.states = [self.state.deepcopy() for i in range(nwalkers)]
        self.lnprob = None
        for i,s in enumerate(self.states):
            shift = 1e-2*self.scales*np.random.normal(size=self.state.Nvars)
            s.shift_params(shift)
    
    def step(self):
        rejections = 0
        n2 = self.nwalkers//2
        g1 = self.states[:n2]
        g2 = self.states[n2:]
        for sg1, sg2 in [[g1,g2],[g2,g1]]:
            params = []
            for s1 in sg1:
                params.append((self, s1, sg2))
            from rebound.interruptible_pool import InterruptiblePool
            pool = InterruptiblePool()
            results = pool.map(update_one,params)
            for r1 in results:
                rejections += r1
                #s1.set_params(r2)
        return rejections
    

    def step_force(self):
        # TODO, following is hackish
        rejections = self.step()
        accepted = self.nwalkers
        return (rejections+accepted)/accepted

    def generate_proposal(self, statei, statej):
        logp, logp_d, logp_dd = statej.get_logp_d_dd(self.obs) 
        Ginv = np.linalg.inv(softabs(logp_dd))
        Ginvsqrt = np.linalg.cholesky(Ginv)   

        epsilon = 0.5
        mu = statei.get_params() + (epsilon)**2 * np.dot(Ginv, logp_d)/2.
        newparams = mu + epsilon * np.dot(Ginvsqrt, np.random.normal(0.,1.,statei.Nvars))
        prop = statei.deepcopy()
        prop.set_params(newparams)
        return prop
    
    def set_scales(self, scales):
        self.scales = np.ones(self.state.Nvars)
        keys = self.state.get_rawkeys()
        for i,k in enumerate(keys):
            if k in scales:
                self.scales[i] = scales[k]


class Ensemble(Mcmc):
    def __init__(self, initial_state, obs, scales, nwalkers=10):
        super(Ensemble,self).__init__(initial_state, obs)
        self.set_scales(scales)
        self.nwalkers = nwalkers
        self.states = [self.state.get_params() for i in range(nwalkers)]
        self.lnprob = None
        for i,s in enumerate(self.states):
            shift = 1e-2*self.scales*np.random.normal(size=self.state.Nvars)
            self.states[i] += shift
        self.sampler = emcee.EnsembleSampler(nwalkers,self.state.Nvars, lnprob, args=[self])

    def step(self):
        self.states, self.lnprob, rstate = self.sampler.run_mcmc(self.states,1,lnprob0=self.lnprob)
        return True

    def set_scales(self, scales):
        self.scales = np.ones(self.state.Nvars)
        keys = self.state.get_rawkeys()
        for i,k in enumerate(keys):
            if k in scales:
                self.scales[i] = scales[k]




class Mh(Mcmc):
    def __init__(self, initial_state, obs):
        super(Mh,self).__init__(initial_state, obs)
        self.step_size = 2e-5

    def generate_proposal(self):
        prop = self.state.deepcopy()
        shift = self.step_size*self.scales*np.random.normal(size=self.state.Nvars)
        prop.shift_params(shift)
        return prop

    def set_scales(self, scales):
        self.scales = np.ones(self.state.Nvars)
        keys = self.state.get_rawkeys()
        for i,k in enumerate(keys):
            if k in scales:
                self.scales[i] = scales[k]

    def step(self):
        logp = self.state.get_logp(self.obs)
        proposal = self.generate_proposal() 
        logp_proposal = proposal.get_logp(self.obs)
        if np.exp(logp_proposal-logp)>np.random.uniform():
            self.state = proposal
            return True
        return False

alpha = 1.
def softabs(hessians):
    lam, Q = np.linalg.eig(-hessians)
    lam_twig = lam*1./np.tanh(alpha*lam)
    H_twig = np.dot(Q,np.dot(np.diag(lam_twig),Q.T))    
    return H_twig

class Smala(Mcmc):
    def __init__(self, initial_state, obs):
        super(Smala,self).__init__(initial_state, obs)
        self.epsilon = 0.5

    def generate_proposal(self):
        logp, logp_d, logp_dd = self.state.get_logp_d_dd(self.obs) 
        Ginv = np.linalg.inv(softabs(logp_dd))
        Ginvsqrt = np.linalg.cholesky(Ginv)   

        mu = self.state.get_params() + (self.epsilon)**2 * np.dot(Ginv, logp_d)/2.
        newparams = mu + self.epsilon * np.dot(Ginvsqrt, np.random.normal(0.,1.,self.state.Nvars))
        prop = self.state.deepcopy()
        prop.set_params(newparams)
        return prop

    def transitionProbability(self,state_from, state_to):
        logp, logp_d, logp_dd = state_from.get_logp_d_dd(self.obs) 
        Ginv = np.linalg.inv(softabs(logp_dd))
        mu = state_from.get_params() + (self.epsilon)**2 * np.dot(Ginv, logp_d)/2.
        return stats.multivariate_normal.logpdf(state_to.get_params(),mean=mu, cov=(self.epsilon)**2*Ginv)
        
    def step(self):
        stateStar = self.generate_proposal()

        q_ts_t = self.transitionProbability(self.state, stateStar)
        q_t_ts = self.transitionProbability(stateStar, self.state)

        if np.exp(stateStar.logp-self.state.logp+q_t_ts-q_ts_t) > np.random.uniform():
            self.state = stateStar
            return True
        return False



