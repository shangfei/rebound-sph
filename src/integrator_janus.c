/**
 * @file    integrator_janus.c
 * @brief   Janus integration scheme.
 * @author  Hanno Rein <hanno@hanno-rein.de>
 * @details This file implements the Janus integration scheme.  
 * Described in Rein & Tamayo 2017.
 * 
 * @section LICENSE
 * Copyright (c) 2017 Hanno Rein, Daniel Tamayo
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
#include "integrator_janus.h"

const static double gamma1 = 0.39216144400731413928;
const static double gamma2 = 0.33259913678935943860;
const static double gamma3 = -0.70624617255763935981;
const static double gamma4 = 0.082213596293550800230;
const static double gamma5 = 0.79854399093482996340;

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

static void drift(struct reb_simulation* r, double dt){
    struct reb_simulation_integrator_janus* ri_janus = &(r->ri_janus);
    const unsigned int N = r->N;
    for(int i=0; i<N; i++){
        ri_janus->p_int[i].x += (REB_PARTICLE_INT_TYPE)(dt*(double)ri_janus->p_int[i].vx) ;
        ri_janus->p_int[i].y += (REB_PARTICLE_INT_TYPE)(dt*(double)ri_janus->p_int[i].vy) ;
        ri_janus->p_int[i].z += (REB_PARTICLE_INT_TYPE)(dt*(double)ri_janus->p_int[i].vz) ;
    }
}

static void kick(struct reb_simulation* r, double dt){
    struct reb_simulation_integrator_janus* ri_janus = &(r->ri_janus);
    const unsigned int N = r->N;
    const double int_scale  = ri_janus->scale;
    for(int i=0; i<N; i++){
        ri_janus->p_int[i].vx += (REB_PARTICLE_INT_TYPE)(int_scale*dt*r->particles[i].ax) ;
        ri_janus->p_int[i].vy += (REB_PARTICLE_INT_TYPE)(int_scale*dt*r->particles[i].ay) ;
        ri_janus->p_int[i].vz += (REB_PARTICLE_INT_TYPE)(int_scale*dt*r->particles[i].az) ;
    }
}

void reb_integrator_janus_part1(struct reb_simulation* r){
    r->gravity_ignore_terms = 0;
    struct reb_simulation_integrator_janus* ri_janus = &(r->ri_janus);
    const unsigned int N = r->N;
    const double dt = r->dt;
    if (ri_janus->allocated_N != N){
        ri_janus->allocated_N = N;
        ri_janus->p_int = realloc(ri_janus->p_int, sizeof(struct reb_particle_int)*N);
        r->ri_janus.is_synchronized = 1;
    }
    
    if (r->ri_janus.is_synchronized==1){
        to_int(ri_janus->p_int, r->particles, N, ri_janus->scale); 
    }
    drift(r,gamma1*dt/2.);
    to_double(r->particles, r->ri_janus.p_int, r->N, r->ri_janus.scale); 
}

void reb_integrator_janus_part2(struct reb_simulation* r){
    struct reb_simulation_integrator_janus* ri_janus = &(r->ri_janus);
    const unsigned int N = r->N;
    const double scale  = ri_janus->scale;
    const double dt = r->dt;
    
    kick(r,gamma1*dt);
    drift(r,(gamma1+gamma2)*dt/2.);
    to_double(r->particles, r->ri_janus.p_int, N, scale); 
    reb_update_acceleration(r);
    kick(r,gamma2*dt);
    drift(r,(gamma2+gamma3)*dt/2.);
    to_double(r->particles, r->ri_janus.p_int, N, scale); 
    reb_update_acceleration(r);
    kick(r,gamma3*dt);
    drift(r,(gamma3+gamma4)*dt/2.);
    to_double(r->particles, r->ri_janus.p_int, N, scale); 
    reb_update_acceleration(r);
    kick(r,gamma4*dt);
    drift(r,(gamma4+gamma5)*dt/2.);
    to_double(r->particles, r->ri_janus.p_int, N, scale); 
    reb_update_acceleration(r);
    kick(r,gamma5*dt);
    drift(r,(gamma5+gamma4)*dt/2.);
    to_double(r->particles, r->ri_janus.p_int, N, scale); 
    reb_update_acceleration(r);
    kick(r,gamma4*dt);
    drift(r,(gamma4+gamma3)*dt/2.);
    to_double(r->particles, r->ri_janus.p_int, N, scale); 
    reb_update_acceleration(r);
    kick(r,gamma3*dt);
    drift(r,(gamma3+gamma2)*dt/2.);
    to_double(r->particles, r->ri_janus.p_int, N, scale); 
    reb_update_acceleration(r);
    kick(r,gamma2*dt);
    drift(r,(gamma2+gamma1)*dt/2.);
    to_double(r->particles, r->ri_janus.p_int, N, scale); 
    reb_update_acceleration(r);
    kick(r,gamma1*dt);
    r->ri_janus.is_synchronized = 0;
    drift(r,gamma1*r->dt/2.);
    if (r->ri_janus.safe_mode){
        reb_integrator_janus_synchronize(r);
    }else{
        // Small overhead here: Always get positions and velocities in floating point at 
        // the end of the timestep.
        to_double(r->particles, r->ri_janus.p_int, r->N, r->ri_janus.scale); 
    }
    r->t += r->dt;
}

void reb_integrator_janus_synchronize(struct reb_simulation* r){
    if(r->ri_janus.is_synchronized==0){
        to_double(r->particles, r->ri_janus.p_int, r->N, r->ri_janus.scale); 
        r->ri_janus.is_synchronized = 1;
    }
}
void reb_integrator_janus_reset(struct reb_simulation* r){
    struct reb_simulation_integrator_janus* const ri_janus = &(r->ri_janus);
    ri_janus->allocated_N = 0;
    ri_janus->safe_mode = 1;
    ri_janus->is_synchronized = 1;
    if (ri_janus->p_int){
        free(ri_janus->p_int);
        ri_janus->p_int = NULL;
    }
}
