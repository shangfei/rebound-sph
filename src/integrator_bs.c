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
#include "integrator.h"
#include "integrator_bs.h"
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define MIN(a, b) ((a) > (b) ? (b) : (a))
	
static const int KMAXX = 8;
static const int IMAXX = 9;
static const double SAFE1 = 0.25;
static const double SAFE2 = 0.7;
static const double SCALMX = 0.1;
static const int nseq[IMAXX] = {2,4,6,8,10,14,16,18};


void reb_integrator_bs_part1(struct reb_simulation* r){
    r->gravity_ignore_terms = 0;
}


static void pzextr(struct reb_simulation* r, const int nv, const int iest, const double xest, double* yest, double* yz, double* dy){
    double** const d = r->ri_bs.d;
    double* const c = r->ri_bs.tmp_c;
    double* const x = r->ri_bs.tmp_x;
    x[iest] = xest;
    for (int j=0;j<nv;j++){
        dy[j] = yz[j] = yest[j];
    }
    if (iest == 0){
        for (int j=0;j<nv;j++){
            d[j][0] = yest[j];
        }
    }else{
        for (int j=0;j<nv;j++){
            c[j] = yest[j];
        }
        for (int k1=0;k1<iest;k1++){
            const double delta = 1./(x[iest-k1-1]-xest);
            const double f1 = xest*delta;
            const double f2 = x[iest-k1-1]*delta;
            for (int j=0;j<nv;j++){
                const double q = d[j][k1];
                d[j][k1] = dy[j];
                const double delta2 = c[j]-q;
                dy[j] = f1*delta2;
                c[j] = f2*delta2;
                yz[j] += dy[j];
            }
        }
        for (int j=0;j<nv;j++){
            d[j][iest] = dy[j];
        }
    }
}

static void update_deriv(struct reb_simulation* r, double* yn, double* yout){
    for (int i=0;i<r->N;i++){
        struct reb_particle* const p = &(r->particles[i]);
        p->x  = yn[i*6+0];
        p->y  = yn[i*6+1];
        p->z  = yn[i*6+2];
        p->vx = yn[i*6+3];
        p->vy = yn[i*6+4];
        p->vz = yn[i*6+5];
    }
    reb_update_acceleration(r); 
    for (int i=0;i<r->N;i++){
        const struct reb_particle p = r->particles[i];
        yout[i*6+0] = p.vx;
        yout[i*6+1] = p.vy;
        yout[i*6+2] = p.vz;
        yout[i*6+3] = p.ax;
        yout[i*6+4] = p.ay;
        yout[i*6+5] = p.az;
    }
}

static void mmid(struct reb_simulation* r,const int nv, double* y, const double xs, const double htot, const int nstep, double* yout){
    const double h = htot/nstep;
    double* const ym = r->ri_bs.tmp_c;
    double* const yn = r->ri_bs.tmp_x;
    const double* const dydx = r->ri_bs.dydx;
    for (int i=0;i<nv;i++){
        ym[i] = y[i];
        yn[i] = y[i] + h*dydx[i];
    }
    r->t = xs + h;
    update_deriv(r,yn,yout);
    const double h2 = 2.0*h;
    for (int n=1;n<nstep;n++){
        for (int i=0;i<nv;i++){
            const double swap = ym[i]+h2*yout[i];
            ym[i] = yn[i];
            yn[i] = swap;
        }
        r->t += h;
        update_deriv(r,yn,yout);
    }
    for (int i=0;i<nv;i++){
        yout[i] = 0.5*(ym[i]+yn[i]+h*yout[i]);
    }
}


void reb_integrator_bs_part2(struct reb_simulation* r){
    int nv = r->N*6;
    if (r->ri_bs.allocated_N<nv){
        r->ri_bs.allocated_N = nv;
        // Matrix d
        if (r->ri_bs.d){
            for(int i=0;i<nv;i++){
                free(r->ri_bs.d[i]);
            }
        }
        r->ri_bs.d = realloc(r->ri_bs.d,nv*sizeof(double *));
        for(int i=0;i<nv;i++){
            r->ri_bs.d[i] = malloc(IMAXX*sizeof(double));
        }
        // Others vectors
        r->ri_bs.tmp_c = realloc(r->ri_bs.tmp_c,nv*sizeof(double));
        r->ri_bs.tmp_x = realloc(r->ri_bs.tmp_x,nv*sizeof(double));
        r->ri_bs.yerr = realloc(r->ri_bs.yerr,nv*sizeof(double));
        r->ri_bs.ysav = realloc(r->ri_bs.ysav,nv*sizeof(double));
        r->ri_bs.y = realloc(r->ri_bs.y,nv*sizeof(double));
        r->ri_bs.dydx = realloc(r->ri_bs.dydx,nv*sizeof(double));
        r->ri_bs.yseq = realloc(r->ri_bs.yseq,nv*sizeof(double));
    }

    double scale_r = 1e-300;
    double scale_v = 1e-300;
    double* const y = r->ri_bs.y;
    double* const dydx = r->ri_bs.dydx;
    for (int i=0;i<r->N;i++){
        struct reb_particle p = r->particles[i];
        y[i*6+0] = p.x;
        y[i*6+1] = p.y;
        y[i*6+2] = p.z;
        y[i*6+3] = p.vx;
        y[i*6+4] = p.vy;
        y[i*6+5] = p.vz;
        dydx[i*6+0] = p.vx;
        dydx[i*6+1] = p.vy;
        dydx[i*6+2] = p.vz;
        dydx[i*6+3] = p.ax;
        dydx[i*6+4] = p.ay;
        dydx[i*6+5] = p.az;
        const double r2 = p.x*p.x + p.y*p.y + p.z*p.z;
        const double v2 = p.vx*p.vx + p.vy*p.vy + p.vz*p.vz;
        scale_r = MAX(scale_r,r2);
        scale_v = MAX(scale_v,v2);
    }
    scale_r = sqrt(scale_r);
    scale_v = sqrt(scale_v);

    double eps1;

    if (r->ri_bs.a == NULL){
        r->ri_bs.a = malloc(IMAXX*sizeof(double));
        r->ri_bs.alf = malloc(KMAXX*sizeof(double *));
        r->ri_bs.err = malloc(IMAXX*sizeof(double));
        for(int i=0;i<KMAXX;i++){
            r->ri_bs.alf[i] = malloc(KMAXX*sizeof(double));
        }
        double* a = r->ri_bs.a;
        eps1 = SAFE1*r->ri_bs.eps;
        a[0] = nseq[0]+1;
        for (int k=0;k<KMAXX;k++){
            a[k+1] = a[k] + nseq[k+1];
        }
        for (int iq=1;iq<KMAXX;iq++){
            for (int k=0;k<iq;k++){
                r->ri_bs.alf[k][iq]=pow(eps1,(a[k+1]-a[iq+1])/((a[iq+1]-a[0]+1.0)*(2*k+3)));
            }
        }
        int kopt;
        for (kopt=1;kopt<KMAXX-1;kopt++){
            if (a[kopt+1] > a[kopt]*r->ri_bs.alf[kopt-1][kopt]){
                break;
            }
        }
        r->ri_bs.kmax = kopt;
        r->ri_bs.kopt = kopt;
    }

    const int kopt = r->ri_bs.kopt;
    const int kmax = r->ri_bs.kmax;
    double* const err = r->ri_bs.err;
    
    double h = r->dt;
    for (int i=0;i<nv;i++){
        r->ri_bs.ysav[i] = y[i];
    }

    int reduct = 0;
    int km;
    int k;
    double t0 = r->t;
    for (;;){
        int exitflag = 0;
        double red;
        for (k=0;k<=kmax;k++){
            mmid(r,nv,r->ri_bs.ysav,t0,h,nseq[k],r->ri_bs.yseq);
            double xest = sqrt(h/nseq[k]);
            pzextr(r,nv,k,xest,r->ri_bs.yseq,y,r->ri_bs.yerr);
            if (k !=0){
                double errmax = 1e-300;
                for (int i=0;i<nv;i++){
                    if (i%6<3){
                        errmax = MAX(errmax,fabs(r->ri_bs.yerr[i]/scale_r));
                    }else{
                        errmax = MAX(errmax,fabs(r->ri_bs.yerr[i]/scale_v));
                    }
                }
                errmax /= r->ri_bs.eps;
                km = k-1;
                err[km] = pow(errmax/SAFE1,1./(2*km+3));
                if (k >= kopt-1 || r->ri_bs.first){
                    if (errmax<1.0){
                        exitflag = 1;
                        break;
                    }
                    if (k == kmax || k == kopt+1){
                        red = SAFE2/err[km];
                        break;
                    }
                    else if (k == kopt && r->ri_bs.alf[kopt-1][kopt] < err[km]){
                        red = 1.0/err[km];
                        break;
                    }
                    else if (kopt == kmax && r->ri_bs.alf[km][kmax-1] < err[km]){
                        red = r->ri_bs.alf[km][kmax-1]*SAFE2/err[km];
                        break;
                    }else if (r->ri_bs.alf[km][kopt]/err[km]) {
                        red = r->ri_bs.alf[km][kopt-1]/err[km];
                        break;
                    }
                }
            }
        }
        if (exitflag) break;
        if (h<r->ri_bs.min_dt || t0==t0+h){
            r->ri_bs.timestep_warning++;
            if (r->ri_bs.timestep_warning == 1 ){
                reb_warning(r, "At least one step of the BS integrator did not converge. This is typically an indication of the timestep being too large.");
            }
            break;
        }
        // restrict scale factor.
        red = MIN(red,0.7);
        red = MAX(red,1e-5);
        h *= red;
        reduct = 1;
    }
    
    // Success 
    
    for (int i=0;i<r->N;i++){
        struct reb_particle *p = &(r->particles[i]);
        p->x  = y[i*6+0];
        p->y  = y[i*6+1];
        p->z  = y[i*6+2];
        p->vx = y[i*6+3];
        p->vy = y[i*6+4];
        p->vz = y[i*6+5];
    }


    r->t = t0 + h;
	r->dt_last_done = h;
    r->ri_bs.first = 0;
    double wrkmin = 1e35;
    double scale;
    for (int kk=0;kk<km;kk++){
        double fact = MAX(err[kk],SCALMX);
        double work = fact*r->ri_bs.a[kk+1];
        if (work < wrkmin){
            scale = fact;
            wrkmin = work;
            r->ri_bs.kopt=kk+1;
        }
    }
    r->dt = h/scale;
    if (r->ri_bs.kopt >= k && r->ri_bs.kopt != kmax && !reduct){
        double fact = MAX(scale/r->ri_bs.alf[r->ri_bs.kopt-1][r->ri_bs.kopt],SCALMX);
        if (r->ri_bs.a[r->ri_bs.kopt+1]*fact <= wrkmin) {
            r->dt = h/fact;
            r->ri_bs.kopt++;
        }
    }

}

void reb_integrator_bs_synchronize(struct reb_simulation* r){
	// Do nothing.
}

void reb_integrator_bs_reset(struct reb_simulation* r){
    // vectors
    free(r->ri_bs.a);
    r->ri_bs.a = NULL;
    free(r->ri_bs.tmp_c);
    r->ri_bs.tmp_c = NULL;
    free(r->ri_bs.tmp_x);
    r->ri_bs.tmp_x = NULL;
    free(r->ri_bs.yerr);
    r->ri_bs.yerr = NULL;
    free(r->ri_bs.ysav);
    r->ri_bs.ysav = NULL;
    free(r->ri_bs.yseq);
    r->ri_bs.yseq = NULL;
    free(r->ri_bs.dydx);
    r->ri_bs.dydx = NULL;
    free(r->ri_bs.y);
    r->ri_bs.y = NULL;
    free(r->ri_bs.err);
    r->ri_bs.err = NULL;
    // Matrix d
    if (r->ri_bs.d){
        for(int i=0;i<r->ri_bs.allocated_N;i++){
            free(r->ri_bs.d[i]);
        }
    }
    free(r->ri_bs.d);
    r->ri_bs.d = NULL;
    if (r->ri_bs.alf){
        for(int i=0;i<KMAXX;i++){
            free(r->ri_bs.alf[i]);
        }
    }
    free(r->ri_bs.alf);
    r->ri_bs.alf = NULL;
    // Constants
    r->ri_bs.eps = 1e-8;
    r->ri_bs.min_dt = 0;
    r->ri_bs.allocated_N = 0;
    r->ri_bs.first = 1;
    r->ri_bs.timestep_warning = 0;
}
