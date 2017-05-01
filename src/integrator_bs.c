/**
 * @file 	integrator.c
 * @brief 	Leap-frog integration scheme.
 * @author 	Hanno Rein <hanno@hanno-rein.de>
 * @details	This file implements the Bulirsch-Stoer integrator.
 *
 * @section 	LICENSE
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
#include "rebound.h"
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define MIN(a, b) ((a) > (b) ? (b) : (a))
	
const int KMAXX = 8;
const int IMAXX = 9;
const double SAFE1 = 0.25;
const double SAFE2 = 0.7;
const double REDMAX = 1e-5;
const double REDMIN = 0.7;
const double TINY = 1e-30;
const double SCALMX = 0.1;


double eps = 1e-15;

const int nseq_d[IMAXX] = {2,4,6,8,10,14,16,18};
const int nseq[IMAXX] = {2,4,6,8,10,14,16,18};


void reb_integrator_bs_part1(struct reb_simulation* r){
    r->gravity_ignore_terms = 0;
}

double* a = NULL;
double** alf = NULL;

void reb_integrator_bs_part2(struct reb_simulation* r){
    a = calloc(IMAXX,sizeof(double));
    double* err = calloc(IMAXX,sizeof(double));
    alf = malloc(KMAXX*sizeof(double *));
    for(int i=0;i<KMAXX;i++){
        alf[i] = calloc(KMAXX,sizeof(double));
    }

    double htry = r->dt;
    int first = 1;
    double epsold = -1.0;
    int kopt, kmax;
    int nv = r->N*6;
    double* yerr = malloc(sizeof(double)*nv);
    double* y = malloc(sizeof(double)*nv);
    double* ysav = malloc(sizeof(double)*nv);
    double* yscal = malloc(sizeof(double)*nv);
    for (int i=0;i<r->N;i++){
        struct reb_particle p = r->particles[i];
        y[i*6+0] = p.x;
        y[i*6+1] = p.y;
        y[i*6+2] = p.z;
        y[i*6+3] = p.vx;
        y[i*6+4] = p.vy;
        y[i*6+5] = p.vz;
        double r = sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
        double v = sqrt(p.vx*p.vx + p.vy*p.vy + p.vz*p.vz);
        yscal[i*6+0] = r;
        yscal[i*6+1] = r;
        yscal[i*6+2] = r;
        yscal[i*6+3] = v;
        yscal[i*6+4] = v;
        yscal[i*6+5] = v;
    }

    double eps1;
    double xnew;

    if (eps != epsold){
        xnew = -1e29;
        eps1 = SAFE1*eps;
        a[0] = nseq[0]+1;
        for (int k=0;k<KMAXX;k++){
            a[k+1] = a[k] + nseq[k+1];
        }
        for (int iq=1;iq<KMAXX;iq++){
            for (int k=0;k<iq;k++){
                alf[k][iq]=pow(eps1,(a[k+1]-a[iq+1])/((a[iq+1]-a[0]+1.0)*(2*k+3)));
            }
        }
        epsold = eps;
        for (kopt=1;kopt<KMAXX-1;kopt++){
            if (a[kopt+1] > a[kopt]*alf[kopt-1][kopt]){
                break;
            }
        }
        kmax = kopt;
    }
    double h = htry;
    for (int i=0;i<nv;i++){
        ysav[i] = y[i];
    }

    int reduct = 0;
    int km;
    int k;
    for (;;){
        int exitflag = 0;
        double red;
        for (k=0;k<=kmax;k++){
            xnew = r->t + h;
            // mmid
            double xest = sqrt(h/nseq[k]);
            //pzextr
            if (k !=0){
                double errmax = TINY;
                for (int i=0;i<nv;i++){
                    errmax = MAX(errmax,fabs(yerr[i]/yscal[i]));
                }
                errmax /= eps;
                km = k-1;
                err[km] = pow(errmax/SAFE1,1./(2*km+3));
                if (k >= kopt-1 || first){
                    if (errmax<1.0){
                        exitflag = 1;
                        break;
                    }
                    if (k == kmax || k == kopt+1){
                        red = SAFE2/err[km];
                        break;
                    }
                    else if (k == kopt && alf[kopt-1][kopt] < err[km]){
                        red = 1.0/err[km];
                        break;
                    }
                    else if (kopt == kmax && alf[km][kmax-1] < err[km]){
                        red = alf[km][kmax-1]*SAFE2/err[km];
                        break;
                    }else if (alf[km][kopt]/err[km]) {
                        red = alf[km][kopt-1]/err[km];
                        break;
                    }
                }
            }
        }
        if (exitflag) break;
        red = MIN(red,REDMIN);
        red = MAX(red,REDMAX);
        h *= red;
        reduct = 1;
    }

    
    for (int i=0;i<r->N;i++){
        struct reb_particle *p = &(r->particles[i]);
        p->x  = y[i*6+0];
        p->y  = y[i*6+1];
        p->z  = y[i*6+2];
        p->vx = y[i*6+3];
        p->vy = y[i*6+4];
        p->vz = y[i*6+5];
    }


    r->t = xnew;
	r->dt_last_done = h;
    first = 0;
    double wrkmin = 1e35;
    double fact;
    double scale;
    for (int kk=0;kk<km;kk++){
        fact = MAX(err[kk],SCALMX);
        double work = fact*a[kk+1];
        if (work < wrkmin){
            scale = fact;
            wrkmin = work;
            kopt=kk+1;
        }
    }
    r->dt = h/scale;
    if (kopt >= k && kopt != kmax && !reduct){
        fact = MAX(scale/alf[kopt-1][kopt],SCALMX);
        if (a[kopt+1]*fact <= wrkmin) {
            r->dt = h/fact;
            kopt++;
        }
    }

}

void reb_integrator_bs_synchronize(struct reb_simulation* r){
	// Do nothing.
}

void reb_integrator_bs_reset(struct reb_simulation* r){
	// Do nothing.
}
