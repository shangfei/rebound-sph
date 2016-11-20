/**
 * @file 	integrator.c
 * @brief 	Levesque-Verlet Bit-Reversible Algorithm.
 * @author 	Hanno Rein <hanno@hanno-rein.de>
 * 
 * @section 	LICENSE
 * Copyright (c) 2016 Hanno Rein
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
#include "rebound.h"

static void rti(double* f){
    double scale = 1e10;
    *f = round(scale*(*f))/scale;
}

void reb_integrator_lvbit_part1(struct reb_simulation* r){
	const int N = r->N;
	const double dt = r->dt;
	struct reb_particle* restrict const particles = r->particles;
    r->gravity_ignore_terms = 0;
    if (r->ri_lvbit.allocated_N < N){
        r->ri_lvbit.p_m = realloc(r->ri_lvbit.p_m, sizeof(struct reb_particle)*N);
        r->ri_lvbit.allocated_N = N;
        r->ri_lvbit.is_synchronized = 1;
    }
    if (r->ri_lvbit.is_synchronized){
        r->ri_lvbit.is_synchronized = 0;
        for (int i=0;i<N;i++){
            r->ri_lvbit.p_m[i].x  = particles[i].x - dt * particles[i].vx;
            r->ri_lvbit.p_m[i].y  = particles[i].y - dt * particles[i].vy;
            r->ri_lvbit.p_m[i].z  = particles[i].z - dt * particles[i].vz;
        
            rti(&(r->ri_lvbit.p_m[i].x));
            rti(&(r->ri_lvbit.p_m[i].y));
            rti(&(r->ri_lvbit.p_m[i].z));
            rti(&(particles[i].x));
            rti(&(particles[i].y));
            rti(&(particles[i].z));
        }
    }
}
void reb_integrator_lvbit_part2(struct reb_simulation* r){
	const int N = r->N;
	struct reb_particle* restrict const particles = r->particles;
	const double dt = r->dt;
	double temp;
    for (int i=0;i<N;i++){
        temp = r->particles[i].x;
		particles[i].x = 2.* particles[i].x - r->ri_lvbit.p_m[i].x + dt*dt * particles[i].ax;
        r->ri_lvbit.p_m[i].x = temp;
        temp = r->particles[i].y;
		particles[i].y = 2.* particles[i].y - r->ri_lvbit.p_m[i].y + dt*dt * particles[i].ay;
        r->ri_lvbit.p_m[i].y = temp;
        temp = r->particles[i].z;
		particles[i].z = 2.* particles[i].z - r->ri_lvbit.p_m[i].z + dt*dt * particles[i].az;
        r->ri_lvbit.p_m[i].z = temp;
        
        rti(&(particles[i].x));
        rti(&(particles[i].y));
        rti(&(particles[i].z));
	}
	r->t+=dt;
	r->dt_last_done = r->dt;
}
	
void reb_integrator_lvbit_synchronize(struct reb_simulation* r){
	const int N = r->N;
	const double dt = r->dt;
	struct reb_particle* restrict const particles = r->particles;
    if (r->ri_lvbit.is_synchronized == 0){
        r->ri_lvbit.is_synchronized = 1;
        for (int i=0;i<N;i++){
            r->particles[i].vx  = (particles[i].x - r->ri_lvbit.p_m[i].x)/dt;
            r->particles[i].vy  = (particles[i].y - r->ri_lvbit.p_m[i].y)/dt;
            r->particles[i].vz  = (particles[i].z - r->ri_lvbit.p_m[i].z)/dt;
        }
    }
}

void reb_integrator_lvbit_reset(struct reb_simulation* r){
	// Do nothing.
}
