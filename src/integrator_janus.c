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

static void to_int(struct reb_particle_int* psi, struct reb_particle* ps, unsigned int N, double int_scale){
    for(unsigned int i=0; i<N; i++){ 
        psi[i].x = ps[i].x*int_scale; 
        psi[i].y = ps[i].y*int_scale; 
        psi[i].z = ps[i].z*int_scale; 
        psi[i].vx = ps[i].vx*int_scale; 
        psi[i].vy = ps[i].vy*int_scale; 
        psi[i].vz = ps[i].vz*int_scale; 
    }
}
static void to_double(struct reb_particle* ps, struct reb_particle_int* psi, unsigned int N, double int_scale){
    for(unsigned int i=0; i<N; i++){ 
        ps[i].x = ((double)psi[i].x)/int_scale; 
        ps[i].y = ((double)psi[i].y)/int_scale; 
        ps[i].z = ((double)psi[i].z)/int_scale; 
        ps[i].vx = ((double)psi[i].vx)/int_scale; 
        ps[i].vy = ((double)psi[i].vy)/int_scale; 
        ps[i].vz = ((double)psi[i].vz)/int_scale; 
    }
}

void reb_integrator_janus_part1(struct reb_simulation* r){
    struct reb_simulation_integrator_janus* ri_janus = &(r->ri_janus);
    const double t = r->t;
    const double dt = r->dt;
    const unsigned int N = r->N;
    const double int_scale  = ri_janus->scale;
    if (ri_janus->allocated_N != N){
        printf("Realloc\n");
        ri_janus->allocated_N = N;
        ri_janus->p_prev = realloc(ri_janus->p_prev, sizeof(struct reb_particle_int)*N);
        ri_janus->p_next = realloc(ri_janus->p_next, sizeof(struct reb_particle_int)*N);
        ri_janus->p_prevrecalc = realloc(ri_janus->p_prevrecalc, sizeof(struct reb_particle_int)*N);
        ri_janus->p_curr = realloc(ri_janus->p_curr, sizeof(struct reb_particle_int)*N);
        
        // Generate cur.
        to_int(ri_janus->p_curr, r->particles, N, int_scale); 
        r->integrator = REB_INTEGRATOR_WHFAST;
        r->dt = -dt;
        reb_step(r);
        r->status = REB_RUNNING;
        r->t = t;
        r->dt = dt;
        to_int(ri_janus->p_prev, r->particles, N, int_scale); 
    }
       
    r->integrator = r->ri_janus.integrator;

    // Backwards  
    r->dt = -dt;
    to_double(r->particles, ri_janus->p_curr, N, int_scale); 
    reb_step(r); 
    to_int(ri_janus->p_prevrecalc, r->particles, N, int_scale);

    // Forward
    r->t = t;
    r->dt = dt; 
    to_double(r->particles, ri_janus->p_curr, N, int_scale); 
    reb_step(r); 
    to_int(ri_janus->p_next, r->particles, N, int_scale);

    for(int i=0; i<N; i++){
        ri_janus->p_next[i].x += ri_janus->p_prev[i].x  - ri_janus->p_prevrecalc[i].x ;
        ri_janus->p_next[i].y += ri_janus->p_prev[i].y  - ri_janus->p_prevrecalc[i].y ;
        ri_janus->p_next[i].z += ri_janus->p_prev[i].z  - ri_janus->p_prevrecalc[i].z ;
        ri_janus->p_next[i].vx += ri_janus->p_prev[i].vx  - ri_janus->p_prevrecalc[i].vx ;
        ri_janus->p_next[i].vy += ri_janus->p_prev[i].vy  - ri_janus->p_prevrecalc[i].vy ;
        ri_janus->p_next[i].vz += ri_janus->p_prev[i].vz  - ri_janus->p_prevrecalc[i].vz ;
    }
    memcpy(ri_janus->p_prev, ri_janus->p_curr, N*sizeof(struct reb_particle_int));
    memcpy(ri_janus->p_curr, ri_janus->p_next, N*sizeof(struct reb_particle_int));

    to_double(r->particles, ri_janus->p_curr, N, int_scale); 
    r->integrator = REB_INTEGRATOR_JANUS;
}
void reb_integrator_janus_flip(struct reb_simulation* r){
    struct reb_simulation_integrator_janus* ri_janus = &(r->ri_janus);
    const unsigned int N = r->N;
    memcpy(ri_janus->p_prevrecalc, ri_janus->p_curr, N*sizeof(struct reb_particle_int));
    memcpy(ri_janus->p_curr, ri_janus->p_prev, N*sizeof(struct reb_particle_int));
    memcpy(ri_janus->p_prev, ri_janus->p_prevrecalc, N*sizeof(struct reb_particle_int));
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
