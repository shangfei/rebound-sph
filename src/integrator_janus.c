/**
 * @file    integrator_janus.c
 * @brief   Janus integration scheme.
 * @author  Hanno Rein <hanno@hanno-rein.de>
 * @details This file implements the WHFast integration scheme.  
 * Described in Rein & Tamayo 2015.
 * 
 * @section LICENSE
 * Copyright (c) 2015 Hanno Rein, Daniel Tamayo
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
#include <sys/time.h>
#include "rebound.h"
#include "particle.h"
#include "tools.h"
#include "gravity.h"
#include "boundary.h"
#include "integrator.h"
#include "integrator_whfast.h"

static void rti(double* f){
    double scale = 1e10;
    *f = round(scale*(*f))/scale;
}
static void round_ps(struct reb_particle* ps, unsigned int N){
    for(unsigned int i=0; i<N; i++){ 
        rti(&(ps[i].x));
        rti(&(ps[i].y));
        rti(&(ps[i].z));
        rti(&(ps[i].vx));
        rti(&(ps[i].vy));
        rti(&(ps[i].vz));
    }        
}

void reb_integrator_janus_part1(struct reb_simulation* r){
    r->integrator = r->ri_janus.integrator;
    struct reb_simulation_integrator_janus* ri_janus = &(r->ri_janus);
    const double t = r->t;
    const unsigned int N = r->N;
    if (ri_janus->allocated_N != N){
        ri_janus->allocated_N = N;
        ri_janus->p_prev = realloc(ri_janus->p_prev, sizeof(struct reb_particle)*N);
        ri_janus->p_prevrecalc = realloc(ri_janus->p_prevrecalc, sizeof(struct reb_particle)*N);
        ri_janus->p_curr = realloc(ri_janus->p_curr, sizeof(struct reb_particle)*N);
    }
    round_ps(ri_janus->p_prev, N); 
    round_ps(r->particles, N); 
    memcpy(ri_janus->p_curr, r->particles, N*sizeof(struct reb_particle)); // cache current
    r->dt = -r->dt;
    reb_step(r); // do backwards step
    round_ps(r->particles, N); // first update to p_prev
    memcpy(ri_janus->p_prevrecalc, r->particles, N*sizeof(struct reb_particle));

    r->dt = -r->dt; // reset particles to values at time t
    r->t = t;
    memcpy(r->particles, ri_janus->p_curr, N*sizeof(struct reb_particle));
    reb_step(r); // do forwards step 


    round_ps(r->particles, N); 
    for(int i=0; i<N; i++){
        r->particles[i].x  += ri_janus->p_prev[i].x  - ri_janus->p_prevrecalc[i].x ;
        r->particles[i].y  += ri_janus->p_prev[i].y  - ri_janus->p_prevrecalc[i].y ;
        r->particles[i].z  += ri_janus->p_prev[i].z  - ri_janus->p_prevrecalc[i].z ;
        r->particles[i].vx += ri_janus->p_prev[i].vx - ri_janus->p_prevrecalc[i].vx;
        r->particles[i].vy += ri_janus->p_prev[i].vy - ri_janus->p_prevrecalc[i].vy;
        r->particles[i].vz += ri_janus->p_prev[i].vz - ri_janus->p_prevrecalc[i].vz;
    }
    round_ps(r->particles, N); 

    memcpy(ri_janus->p_prev, ri_janus->p_curr, N*sizeof(struct reb_particle));
    r->integrator = REB_INTEGRATOR_JANUS;
}
void reb_integrator_janus_part2(struct reb_simulation* r){
}
void reb_integrator_janus_synchronize(struct reb_simulation* r){
}
void reb_integrator_janus_reset(struct reb_simulation* r){
    struct reb_simulation_integrator_janus* const ri_janus = &(r->ri_janus);
    ri_janus->allocated_N = 0;
    if (ri_janus->p_prev){
        free(ri_janus->p_prev);
        ri_janus->p_prev = NULL;
    }
    if (ri_janus->p_prevrecalc){
        free(ri_janus->p_prevrecalc);
        ri_janus->p_prevrecalc = NULL;
    }
    if (ri_janus->p_curr){
        free(ri_janus->p_curr);
        ri_janus->p_curr = NULL;
    }
}
