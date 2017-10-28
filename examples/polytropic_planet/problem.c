/**
 * Polytropic planet
 *
 * This example integrates a densely packed planetary system 
 * which becomes unstable on a timescale of only a few orbits. The IAS15 
 * integrator with adaptive timestepping is used. This integrator 
 * automatically decreases the timestep whenever a close 
 * encounter happens. IAS15 is very high order and ideally suited for the 
 * detection of these kind of encounters.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "rebound.h"

void heartbeat(struct reb_simulation* r);

int main(int argc, char* argv[]){
	struct reb_simulation* r = reb_create_simulation();
	// r->dt 		= 0.01*2.*M_PI;		// initial timestep
	r->gravity	= REB_GRAVITY_TREE;
	r->boundary	= REB_BOUNDARY_OPEN;
	r->opening_angle2	= 1.e-2; // 1.5/10.;		// This constant determines the accuracy of the tree code gravity estimate.
	r->G 		= 6.674e-8;		
	r->softening 	= 0.02;		// Gravitational softening length
	r->dt 		= 0.5; //3e-2*1000;		// Timestep
	r->gamma    = 2.;
	r->initSPH	= 1;
	const double boxsize = 3e10;
	r->integrator 	= REB_INTEGRATOR_LEAPFROG;
	r->heartbeat  	= heartbeat;
	// r->usleep	= 100;		// Slow down integration (for visualization only)
	reb_configure_box(r,boxsize,1,1,1);
	
	double total_mass = 1.898e30;// 4*M_PI*M_PI;
	int N = 2000;
	const double K = 2.6e12;
	double alpha = sqrt(K/2./M_PI/r->G);
	int Nbin = 50;
	double dxi = M_PI/( (double) Nbin);
	double mp = total_mass / (double)N;
	r->m = mp;
	double R = 6.99e9;
	double smoothing_length = R/ 5.; //sqrt((double) N)*500.; 
	double rhoc = 5.;
	int n = 0;

	for (int i=0;i<Nbin;i++){
		// if (n>=N) break;
		struct reb_particle pt = {0};
		// double a	= reb_random_powerlaw(boxsize/2./10.,boxsize/2./2./1.2,-1.5);
		double xi1 = dxi*i;
		double xi2 = dxi * (i+1);
		// printf("xi1 and xi2 are %f %f\n", xi1, xi2);
		int N_in_bin =  (int)round((sin(xi2)-xi2*cos(xi2)-sin(xi1)+xi1*cos(xi1))/M_PI*(double)N);
		// printf("Nbin is %i and N_in_bin is %i \n", i, N_in_bin);
		for (int j=0;j<N_in_bin;j++){
			double phi 	= reb_random_uniform(0,2.*M_PI);
			double cos_theta = reb_random_uniform(-1, 1);
			double xi = reb_random_uniform(xi1, xi2);
			pt.x = alpha * xi * sin(acos(cos_theta))*cos(phi);
			pt.y = alpha * xi * sin(acos(cos_theta))*sin(phi);
			pt.z = alpha * xi * cos_theta;
			pt.m = mp;
			pt.h = smoothing_length;
			double rho = sin(xi)/xi * rhoc;
			// double pressure = K*rho*rho;
			// pt.e = K*rho;
			pt.p = K*rho*rho;
			reb_add(r, pt);
		}
		n++;
	}
	reb_move_to_com(r);		// This makes sure the planetary systems stays within the computational domain and doesn't drift.
	reb_integrate(r, INFINITY);
}

void heartbeat(struct reb_simulation* r){
	char* checkfile[16];
	if (reb_output_check(r, 10.*M_PI)){
		reb_output_ascii(r, "sph.txt");  
		reb_output_timing(r, 0);
		// reb_move_to_com(r);
	}	
	if (reb_output_check(r, 3600.*M_PI)){
		sprintf(checkfile, "checkpoint%04d.h5", (int)round(r->t/20./M_PI));
		reb_output_hdf5(r, checkfile);
	}
}
