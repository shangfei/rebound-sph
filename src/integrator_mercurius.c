/**
 * @file    integrator_mercurius.c
 * @brief   MERCURIUS, an improved version of John Chambers' MERCURY algorithm
 * @author  Hanno Rein
 * 
 * @section LICENSE
 * Copyright (c) 2017 Hanno Rein 
 *
 * This file is part of rebound.
 *
 * rebound is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * rebound is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with rebound.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "rebound.h"
#include "output.h"
#include "integrator_mercurius.h"
#include "integrator_ias15.h"
#include "integrator_whfast.h"
#include "integrator_whfasthelio.h"

double reb_integrator_mercurius_K(double r, double rcrit){
    double y = (r-0.1*rcrit)/(0.9*rcrit);
    if (y<0.){
        return 0.;
    }
    if (y>1.){
        return 1.;
    }
    return y*y/(2.*y*y-2.*y+1.);
}

static void reb_mercurius_ias15step(struct reb_simulation* const r, const double _dt){
    struct reb_simulation_integrator_mercurius* ri_mercurius = &(r->ri_mercurius);
	struct reb_particle* const particles = r->particles;
    const int N = r->N;
	const int N_active = ((r->N_active==-1)?N:r->N_active);
    ri_mercurius->encounterN = 0;
    for (int i=0; i<N; i++){
        ri_mercurius->encounterIndicies[i] = 0;
    }
    for (int i=1; i<N; i++){
    for (int j=i+1; j<N_active; j++){
        const double dx = particles[i].x - particles[j].x;
        const double dy = particles[i].y - particles[j].y;
        const double dz = particles[i].z - particles[j].z;
        const double _r = sqrt(dx*dx + dy*dy + dz*dz);
        const double _K = reb_integrator_mercurius_K(_r,0.1);
        if (_K<1.){
            // encounter
            ri_mercurius->encounterN++;
            ri_mercurius->encounterIndicies[i] = i;
            ri_mercurius->encounterIndicies[j] = j;
        }
    }
    }
    if (ri_mercurius->encounterN==0){
        return; // Nothing to do.
    }
    if (ri_mercurius->allocatedias15N<=ri_mercurius->encounterN){
        ri_mercurius->allocatedias15N = ri_mercurius->encounterN;
        ri_mercurius->ias15particles = realloc(ri_mercurius->ias15particles, sizeof(struct reb_particle)*ri_mercurius->encounterN);
    }
    struct reb_particle* ias15p = ri_mercurius->ias15particles;
    struct reb_particle* p = r->particles;

    int j = 0;
    for (int i=0; i<N; i++){
        if(ri_mercurius->encounterIndicies[i]>0){
            ias15p[j] = p[i];
            j++;
        }
    }

    // Swap
    r->particles = ias15p;
    r->N = j;
    //ias15

    r->particles = p;
    r->N = N;

}

static void reb_mercurius_jumpstep(const struct reb_simulation* const r, double _dt){
    const int N = r->N;
    struct reb_particle* const p_h = r->ri_whfasthelio.p_h;
    const double m0 = r->particles[0].m;
    double px=0, py=0, pz=0;
    for(int i=1;i<N;i++){
        const double m = r->particles[i].m;
        px += m * p_h[i].vx / (m0+m);
        py += m * p_h[i].vy / (m0+m);
        pz += m * p_h[i].vz / (m0+m);
    }
    for(int i=1;i<N;i++){
        const double m = r->particles[i].m;
        p_h[i].x += _dt * (px - (m * p_h[i].vx / (m0+m)) );
        p_h[i].y += _dt * (py - (m * p_h[i].vy / (m0+m)) );
        p_h[i].z += _dt * (pz - (m * p_h[i].vz / (m0+m)) );
    }
}

static void reb_mercurius_interactionstep(const struct reb_simulation* const r, const double _dt){
    struct reb_particle* particles = r->particles;
    const int N = r->N;
    struct reb_particle* const p_h = r->ri_whfasthelio.p_h;
    const double m0 = r->particles[0].m;   
    for (unsigned int i=1;i<N;i++){
        const double m = r->particles[i].m;  
        p_h[i].vx += _dt*particles[i].ax*(m+m0)/m0;
        p_h[i].vy += _dt*particles[i].ay*(m+m0)/m0;
        p_h[i].vz += _dt*particles[i].az*(m+m0)/m0;
    }
}

static void reb_mercurius_keplerstep(const struct reb_simulation* const r, const double _dt){
    const int N = r->N;
    struct reb_particle* const p_h = r->ri_whfasthelio.p_h;
    const struct reb_simulation_integrator_mercurius* ri_mercurius = &(r->ri_mercurius);
    const double m0 = r->particles[0].m;
#pragma omp parallel for
    for (unsigned int i=1;i<N;i++){
        if (ri_mercurius->encounterIndicies[i]==0){
            kepler_step(r, p_h, r->G*(p_h[i].m + m0), i, _dt);
        }
    }
    p_h[0].x += _dt*p_h[0].vx;
    p_h[0].y += _dt*p_h[0].vy;
    p_h[0].z += _dt*p_h[0].vz;
}
            

					

void reb_integrator_mercurius_part1(struct reb_simulation* r){
    if (r->var_config_N){
        reb_exit("Mercurius does currently not work with variational equations.");
    }
    r->gravity = REB_GRAVITY_MERCURIUS;
    r->gravity_ignore_terms = 2;


}

void reb_integrator_mercurius_part2(struct reb_simulation* const r){
    struct reb_particle* restrict const particles = r->particles;
    struct reb_simulation_integrator_mercurius* const ri_mercurius = &(r->ri_mercurius);
    struct reb_simulation_integrator_whfasthelio* const ri_whfasthelio = &(r->ri_whfasthelio);
    const int N = r->N;
    if (ri_mercurius->allocatedN<=N){
        ri_mercurius->allocatedN = N;
        ri_mercurius->encounterIndicies = realloc(ri_mercurius->encounterIndicies, sizeof(unsigned int)*N);
    }
    if (ri_whfasthelio->allocated_N != N){
        ri_whfasthelio->allocated_N = N;
        ri_whfasthelio->p_h = realloc(ri_whfasthelio->p_h,sizeof(struct reb_particle)*N);
        ri_whfasthelio->recalculate_heliocentric_this_timestep = 1;
    }

    if (ri_whfasthelio->recalculate_heliocentric_this_timestep == 1){
        ri_whfasthelio->recalculate_heliocentric_this_timestep = 0;
        reb_transformations_inertial_to_democratic_heliocentric_posvel(particles, ri_whfasthelio->p_h, N);
    }
    
    
    
    reb_mercurius_interactionstep(r,r->dt/2.);
    reb_mercurius_jumpstep(r,r->dt/2.);
    reb_mercurius_keplerstep(r,r->dt);
    reb_mercurius_ias15step(r,r->dt);
    reb_mercurius_jumpstep(r,r->dt/2.);
    reb_transformations_democratic_heliocentric_to_inertial_posvel(particles, ri_whfasthelio->p_h, N);
    reb_mercurius_interactionstep(r,r->dt/2.);
    
    
    
    

    r->t+=r->dt;
    r->dt_last_done = r->dt;
}

void reb_integrator_mercurius_synchronize(struct reb_simulation* r){
}

void reb_integrator_mercurius_reset(struct reb_simulation* r){
    r->ri_mercurius.mode = 0;
}

