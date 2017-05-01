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
#include "integrator.h"
#include "gravity.h"
#include "tools.h"
#include "integrator_mercurius.h"
#include "integrator_ias15.h"
#include "integrator_bs.h"
#include "integrator_leapfrog.h"
#include "integrator_whfast.h"
#define MIN(a, b) ((a) > (b) ? (b) : (a))    ///< Returns the minimum of a and b

double reb_integrator_mercurius_K(double r, double rcrit){
    double y = (r-0.1*rcrit)/(0.9*rcrit);
    if (y<0.){
        return 0.;
    }
    if (y>1.){
        return 1.;
    }


    return 0.5*(15./8.*(2.*y - 1.) - 5./4.*powf(2.*y - 1.,3) + 3./8.*pow(2.*y - 1.,5) + 1.);
    //return y*y/(2.*y*y-2.*y+1.);
}
int count = 0;
double e0;
void debug(struct reb_simulation* r){
    
    FILE *fp;
    if (count==0){
        fp=fopen("close.txt", "w");
        e0 = reb_tools_energy(r);
    }else{
        fp=fopen("close.txt", "a+");
    }
    count++;
    double e = reb_tools_energy(r);
    double rmin = 10000;
    for (int i=0;i<r->N;i++){
        for (int j=0;j<r->N;j++){
            if (i==j) continue;
            double dx = r->particles[i].x - r->particles[j].x;
            double dy = r->particles[i].y - r->particles[j].y;
            double dz = r->particles[i].z - r->particles[j].z;
            rmin = MIN(rmin, sqrt(dx*dx+dy*dy+dz*dz));
        }
    }
    int eN = r->ri_mercurius.encounterN;
    fprintf(fp, "%f %f %e %d %f %f\n",r->t/365.,rmin/r->ri_mercurius.rcrit,fabs((e-e0)/e0),eN,reb_integrator_mercurius_K(rmin,r->ri_mercurius.rcrit),reb_integrator_mercurius_dKdr(rmin,r->ri_mercurius.rcrit));
    fclose(fp);
     
    if (fabs((e-e0)/e0)>1e-3){
        exit(0);
    }
}

double reb_integrator_mercurius_dKdr(double r, double rcrit){
    double y = (r-0.1*rcrit)/(0.9*rcrit);
    if (y<0. || y >1.){
        return 0.;
    }
    return 30.*(1.-y)*(1.-y)*y*y;
    //const double den = 2.*y*y-2.*y+1;
    //return -1./(0.9*rcrit)* 2.*(y-1.)*y/(den*den);
}

static void reb_mercurius_ias15step(struct reb_simulation* const r, const double _dt){
    struct reb_simulation_integrator_mercurius* ri_mercurius = &(r->ri_mercurius);
	struct reb_particle* const p_hold = ri_mercurius->p_hold;
	struct reb_particle* const p_h = ri_mercurius->p_h;
    const int N = r->N;
    if (ri_mercurius->encounterN==0){
        return; // Nothing to do.
    }
    if (ri_mercurius->allocatedias15N<=ri_mercurius->encounterN){
        ri_mercurius->allocatedias15N = ri_mercurius->encounterN;
        ri_mercurius->ias15particles = realloc(ri_mercurius->ias15particles, sizeof(struct reb_particle)*ri_mercurius->encounterN);
    }
    struct reb_particle* ias15p = ri_mercurius->ias15particles;

    int j = 0;
    for (int i=0; i<N; i++){
        if(ri_mercurius->encounterIndicies[i]>0){
            ias15p[j] = p_hold[i];
            j++;
        }
    }

    // Swap
    struct reb_particle* old_p = r->particles;
    r->particles = ias15p;
    r->N = j;
    r->ri_mercurius.mode = 1;
    
    // run
    const double old_dt = r->dt;
    const double old_t = r->t;
    double t_needed = r->t + _dt; 
    r->ri_ias15.min_dt = 0.001*r->dt;
    r->dt *= 0.512385;
    reb_integrator_bs_reset(r);
    while(r->t < t_needed && fabs(r->dt/old_dt)>1e-12 ){
    reb_integrator_ias15_reset(r);
        //:aprintf("%e %e %e\n", r->dt, old_t, r->t - (old_t+_dt));
        //reb_integrator_leapfrog_part1(r);
        reb_update_acceleration(r);
        reb_integrator_bs_part2(r);
        //reb_integrator_leapfrog_part2(r);
        if (r->t+r->dt >  t_needed){
            r->dt = t_needed-r->t;
        }
    }
    if (r->t-t_needed!=0.){
    printf("%f\n",r->t-t_needed);
    }
    r->t = old_t;
    r->dt = old_dt;

    // swap 
    j=0;
    for (int i=0; i<N; i++){
        if(ri_mercurius->encounterIndicies[i]>0){
            p_h[i] = ias15p[j];
            j++;
        }
    }


    r->ri_mercurius.mode = 0;
    r->particles = old_p;
    r->N = N;

}

static void reb_mercurius_jumpstep(const struct reb_simulation* const r, double _dt){
    const int N = r->N;
    struct reb_particle* const p_h = r->ri_mercurius.p_h;
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
    struct reb_particle* const p_h = r->ri_mercurius.p_h;
    for (unsigned int i=1;i<N;i++){
        p_h[i].vx += _dt*particles[i].ax;
        p_h[i].vy += _dt*particles[i].ay;
        p_h[i].vz += _dt*particles[i].az;
    }
}

static void reb_mercurius_keplerstep(const struct reb_simulation* const r, const double _dt){
    const int N = r->N;
    struct reb_particle* const p_h = r->ri_mercurius.p_h;
    const double m0 = r->particles[0].m;
#pragma omp parallel for
    for (unsigned int i=1;i<N;i++){
        kepler_step(r, p_h, r->G*(p_h[i].m + m0), i, _dt);
    }
}

static void reb_mercurius_comstep(const struct reb_simulation* const r, const double _dt){
    struct reb_particle* const p_h = r->ri_mercurius.p_h;
    p_h[0].x += _dt*p_h[0].vx;
    p_h[0].y += _dt*p_h[0].vy;
    p_h[0].z += _dt*p_h[0].vz;
}
            
static void reb_mercurius_predict_encounters(struct reb_simulation* const r){
    struct reb_simulation_integrator_mercurius* ri_mercurius = &(r->ri_mercurius);
	struct reb_particle* const p_hn = ri_mercurius->p_h;
	struct reb_particle* const p_ho = ri_mercurius->p_hold;
    const int N = r->N;
    const double dt = r->dt;
    const double rcrit = ri_mercurius->rcrit;
    ri_mercurius->encounterN = 0;
    for (int i=0; i<N; i++){
        ri_mercurius->encounterIndicies[i] = 0;
    }
    for (int i=1; i<N; i++){
    for (int j=i+1; j<N; j++){
        const double dxn = p_hn[i].x - p_hn[j].x;
        const double dyn = p_hn[i].y - p_hn[j].y;
        const double dzn = p_hn[i].z - p_hn[j].z;
        const double dvxn = p_hn[i].vx - p_hn[j].vx;
        const double dvyn = p_hn[i].vy - p_hn[j].vy;
        const double dvzn = p_hn[i].vz - p_hn[j].vz;
        const double rn = sqrt(dxn*dxn + dyn*dyn + dzn*dzn);
        const double dxo = p_ho[i].x - p_ho[j].x;
        const double dyo = p_ho[i].y - p_ho[j].y;
        const double dzo = p_ho[i].z - p_ho[j].z;
        const double dvxo = p_ho[i].vx - p_ho[j].vx;
        const double dvyo = p_ho[i].vy - p_ho[j].vy;
        const double dvzo = p_ho[i].vz - p_ho[j].vz;
        const double ro = sqrt(dxo*dxo + dyo*dyo + dzo*dzo);

        const double drndt = dxn*dvxn/rn+dyn*dvyn/rn+dzn*dvzn/rn;
        const double drodt = dxo*dvxo/ro+dyo*dvyo/ro+dzo*dvzo/ro;

        const double a = 6.*(ro-rn)+3.*dt*(drodt+drndt); 
        const double b = 6.*(rn-ro)-2.*dt*(2.*drodt+drndt); 
        const double c = dt*drodt; 

        double rmin = MIN(rn,ro);

        const double s = b*b-4.*a*c;
        const double sr = sqrt(s);
        const double tmin1 = (-b + sr)/(2.*a); 
        const double tmin2 = (-b - sr)/(2.*a); 
        if (tmin1>0. && tmin1<1.){
            const double rmin1 = (1.-tmin1)*(1.-tmin1)*(1.+2.*tmin1)*ro
                                 + tmin1*tmin1*(3.-2.*tmin1)*rn
                                 + tmin1*(1.-tmin1)*(1.-tmin1)*dt*drodt
                                 - tmin1*tmin1*(1.-tmin1)*dt*drndt;
            rmin = MIN(rmin,rmin1);
        }
        if (tmin2>0. && tmin2<1.){
            const double rmin2 = (1.-tmin2)*(1.-tmin2)*(1.+2.*tmin2)*ro
                                 + tmin2*tmin2*(3.-2.*tmin2)*rn
                                 + tmin2*(1.-tmin2)*(1.-tmin2)*dt*drodt
                                 - tmin2*tmin2*(1.-tmin2)*dt*drndt;
            rmin = MIN(rmin,rmin2);
        }


        
        if (rmin< 1.1*rcrit){
            // encounter
            // TODO: Need to predict encounter during step.
            if (ri_mercurius->encounterIndicies[i]==0){
                ri_mercurius->encounterIndicies[i] = i;
                ri_mercurius->encounterN++;
            }
            if (ri_mercurius->encounterIndicies[j]==0){
                ri_mercurius->encounterIndicies[j] = j;
                ri_mercurius->encounterN++;
            }
        }
    }
    }

}

					

void reb_integrator_mercurius_part1(struct reb_simulation* r){
    if (r->t > 160*365.){
        //reb_output_binary(r,"out.bin");
        //exit(0);
    }
    debug(r);
    if (r->var_config_N){
        reb_exit("Mercurius does currently not work with variational equations.");
    }
    r->gravity = REB_GRAVITY_MERCURIUS;
    r->ri_mercurius.mode = 0;
    r->ri_mercurius.m0 = r->particles[0].m;
}

void reb_integrator_mercurius_part2(struct reb_simulation* const r){
    struct reb_particle* restrict const particles = r->particles;
    struct reb_simulation_integrator_mercurius* const ri_mercurius = &(r->ri_mercurius);
    const int N = r->N;
    if (ri_mercurius->allocatedN<=N){
        ri_mercurius->allocatedN = N;
        ri_mercurius->encounterIndicies = realloc(ri_mercurius->encounterIndicies, sizeof(unsigned int)*N);
        ri_mercurius->p_h = realloc(ri_mercurius->p_h,sizeof(struct reb_particle)*N);
        ri_mercurius->p_hold = realloc(ri_mercurius->p_hold,sizeof(struct reb_particle)*N);
        reb_transformations_inertial_to_democratic_heliocentric_posvel(particles, ri_mercurius->p_h, N);
    }
    
   
    
    
    reb_transformations_democratic_heliocentric_to_inertial_posvel(particles, ri_mercurius->p_h, N);
    reb_calculate_acceleration(r);
    reb_mercurius_interactionstep(r,r->dt/2.);
    reb_mercurius_jumpstep(r,r->dt/2.);
   
   
    memcpy(ri_mercurius->p_hold,ri_mercurius->p_h,N*sizeof(struct reb_particle));
    reb_mercurius_comstep(r,r->dt);
    reb_mercurius_keplerstep(r,r->dt);
    
    reb_mercurius_predict_encounters(r);
    
    reb_mercurius_ias15step(r,r->dt);
    
    
    reb_mercurius_jumpstep(r,r->dt/2.);
    reb_transformations_democratic_heliocentric_to_inertial_posvel(particles, ri_mercurius->p_h, N);
    reb_calculate_acceleration(r);
    reb_mercurius_interactionstep(r,r->dt/2.);
    
    
    reb_transformations_democratic_heliocentric_to_inertial_posvel(particles, ri_mercurius->p_h, N);
    r->t+=r->dt;
    r->dt_last_done = r->dt;
}

void reb_integrator_mercurius_synchronize(struct reb_simulation* r){
    struct reb_particle* restrict const particles = r->particles;
    struct reb_simulation_integrator_mercurius* const ri_mercurius = &(r->ri_mercurius);
    const int N = r->N;
    reb_transformations_democratic_heliocentric_to_inertial_posvel(particles, ri_mercurius->p_h, N);
}

void reb_integrator_mercurius_reset(struct reb_simulation* r){
    r->ri_mercurius.mode = 0;
}

