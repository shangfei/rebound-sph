/**
 * @file 	eos.c
 * @brief 	Equation of state
 * @author 	Shangfei Liu <shangfei.liu@gmail.com>
 *
 * @details
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include "particle.h"
#include "rebound.h"
#include "tree.h"
#include "boundary.h"
#define MAX(a, b) ((a) > (b) ? (a) : (b))    ///< Returns the maximum of a and b
#define MIN(a, b) ((a) > (b) ? (b) : (a))	 ///< Returns the minimum of a and b

void reb_eos_init(struct reb_simulation* const r){
	switch (r->eos) {
		case REB_EOS_DUMMY:
			break;
		case REB_EOS_POLYTROPE:
			r->eos_polytrope.gamma = 1.+1./r->eos_polytrope.n;
			break;
		case REB_EOS_GAMMA_LAW:
			break;
		case REB_EOS_ISOTHERMAL:
			break;
		case REB_EOS_TILLOTSON:
			break;

	}
}

static void reb_eos_dummy (const struct reb_simulation* const r, const int pt){
	struct reb_particle* const particles = r->particles;
	particles[pt].p = 0.;
}


static void reb_eos_polytrope (const struct reb_simulation* const r, const int pt){
	struct reb_particle* const particles = r->particles;
	particles[pt].p = r->eos_polytrope.K * pow(particles[pt].rhoi, r->eos_polytrope.gamma);
}

static void reb_eos_gammalaw (const struct reb_simulation* const r, const int pt){
	struct reb_particle* const particles = r->particles;	
	particles[pt].p = (r->eos_gammalaw.gamma - 1.) * particles[pt].rhoi * particles[pt].e;
}

static void reb_eos_isothermal(const struct reb_simulation* const r, const int pt){
	struct reb_particle* const particles = r->particles;
	particles[pt].p = particles[pt].cs*particles[pt].cs*particles[pt].rhoi;
}

static void reb_eos_tillotson(const struct reb_simulation* const r, const int pt){
	static double rho0 = 2.7; 
	static double a = 2.67e11;
	static double b = 2.67e11;
	static double e0 = 4.87e12;
	static double eiv = 4.72e10;
	static double ecv = 1.82e11;
	static double aa = 0.5;
	static double bb = 1.5;
	static double alpha = 5.0;
	static double beta = 5.0; 
	struct reb_particle* const particles = r->particles;
	double etat = particles[pt].rhoi/rho0;
	double mu = etat - 1.0;
	if ((particles[pt].e <= eiv) || (particles[pt].rhoi >= rho0)) {
		particles[pt].p = (aa+bb/(particles[pt].e/e0/etat/etat+1))*particles[pt].rhoi*particles[pt].e + a*mu + b*mu*mu;
		if ((particles[pt].rhoi < rho0) && (particles[pt].p < aa*particles[pt].rhoi*particles[pt].e)) {
			particles[pt].p = aa*particles[pt].rhoi*particles[pt].e;
			// gammac = 
		} else {
			// gammac = 
		}
	} else if ((particles[pt].e >= ecv) && (particles[pt].rhoi < rho0)) {
		particles[pt].p = aa*particles[pt].rhoi*particles[pt].e + (bb*particles[pt].rhoi/(1/e0/etat/etat+1/particles[pt].e) + a*mu*exp(-beta*(rho0/particles[pt].rhoi))) * exp(-alpha*(rho0/particles[pt].rhoi-1)*(rho0/particles[pt].rhoi-1));
		// gammac = 
	} else if ((particles[pt].rhoi < rho0) && (particles[pt].e > eiv) && (particles[pt].e < ecv)) {
		particles[pt].p = ((aa+bb/(particles[pt].e/e0/etat/etat+1))*particles[pt].rhoi*particles[pt].e + a*mu + b*mu*mu) * (ecv-particles[pt].e)/(ecv-eiv) + (aa*particles[pt].rhoi*particles[pt].e + (bb*particles[pt].rhoi/(1/e0/etat/etat+1/particles[pt].e) + a*mu*exp(-beta*(rho0/particles[pt].rhoi-1))) * exp(-alpha*(rho0/particles[pt].rhoi-1)*(rho0/particles[pt].rhoi-1))) * (particles[pt].e-eiv)/(ecv-eiv);
		// gammac = 
	}
}

void reb_calculate_internal_energy_for_sph_particle(struct reb_simulation* r, int pt){
	struct reb_particle* const particles = r->particles;
	double rhoratio = particles[pt].rhoi/particles[pt].rho;
	particles[pt].e = particles[pt].p/(r->hydro.gamma-1.)/particles[pt].rho * rhoratio *rhoratio;
}

void reb_eos (const struct reb_simulation* const r, const int pt){
	switch (r->eos) {
		case REB_EOS_DUMMY:
			reb_eos_dummy(r, pt);
			break;
		case REB_EOS_POLYTROPE:
			reb_eos_polytrope(r, pt);
			break;
		case REB_EOS_GAMMA_LAW:
			reb_eos_gammalaw(r, pt);
			break;
		case REB_EOS_ISOTHERMAL:
			reb_eos_isothermal(r, pt);
			break;
		case REB_EOS_TILLOTSON:
			reb_eos_tillotson(r, pt);
			break;
	}
}
