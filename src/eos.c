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
		case REB_EOS_NONE:
		break;
		case REB_EOS_POLYTROPE:
			r->eos_polytrope.gamma = 1.+1./r->eos_polytrope.n;
		case REB_EOS_GAMMA_LAW:
		break;
	}
}

static void reb_eos_polytrope (const struct reb_simulation* const r, const int pt){
	struct reb_particle* const particles = r->particles;
	particles[pt].p = r->eos_polytrope.K * pow(particles[pt].rhoi, r->eos_polytrope.gamma);
}

static void reb_eos_gammalaw (const struct reb_simulation* const r, const int pt){
	struct reb_particle* const particles = r->particles;	
	particles[pt].p = (r->eos_gammalaw.gamma - 1.) * particles[pt].rhoi * particles[pt].e;
}

void reb_calculate_internal_energy_for_sph_particle(struct reb_simulation* r, int pt){
	struct reb_particle* const particles = r->particles;
	double rhoratio = particles[pt].rhoi/particles[pt].rho;
	particles[pt].e = particles[pt].p/(r->hydro.gamma-1.)/particles[pt].rho * rhoratio *rhoratio;
}

void reb_eos (const struct reb_simulation* const r, const int pt){
	switch (r->eos) {
		case REB_EOS_NONE:
			break;
		case REB_EOS_POLYTROPE:{
			reb_eos_polytrope(r, pt);
			break;
		}
		case REB_EOS_GAMMA_LAW:{
			reb_eos_gammalaw(r, pt);
			break;
		}
	}	
}
