/**
 * Tidally disrupted Jupiter
 *
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
	r->dt 		= 0.05; //3e-2*1000;		// Timestep
	r->initSPH	= 1;
	const double boxsize = 1.e13;
	r->integrator 	= REB_INTEGRATOR_LEAPFROG;
	// EOS
	r->eos		= REB_EOS_POLYTROPE;
	r->eos_polytrope.n = 1;
	r->eos_polytrope.K = 2.6e12; // dyne g^-2 cm^4

	r->hydro.nnmin = 45;
	r->hydro.nnmax = 55;

	r->heartbeat  	= heartbeat;
	reb_configure_box(r,boxsize,1,1,1);
	
	double jupiter_mass = 1.898e30;// 4*M_PI*M_PI;
	int N = 2500;
	double alpha = sqrt(r->eos_polytrope.K/2./M_PI/r->G);
	int Nbin = 50;
	double dxi = M_PI/( (double) Nbin);
	double mp = jupiter_mass / (double)N;
	r->m = mp;
	double R = 6.99e9;
	double smoothing_length = R/ 5.; //sqrt((double) N)*500.; 
	double rhoc = 5.;
	int n = 0;

	for (int i=0;i<Nbin;i++){
		struct reb_particle pt = {0};
		double xi1 = dxi*i;
		double xi2 = dxi * (i+1);
		int N_in_bin =  (int)round((sin(xi2)-xi2*cos(xi2)-sin(xi1)+xi1*cos(xi1))/M_PI*(double)N);
		for (int j=0;j<N_in_bin;j++){
			double phi 	= reb_random_uniform(0,2.*M_PI);
			double cos_theta = reb_random_uniform(-1, 1);
			double xi = reb_random_uniform(xi1, xi2);
			pt.x = alpha * xi * sin(acos(cos_theta))*cos(phi) - 9.486838571279353E10;
			pt.y = alpha * xi * sin(acos(cos_theta))*sin(phi) - 5.169034544361552E9;
			pt.z = alpha * xi * cos_theta;
			pt.vx = 3.660946159165656E7;
			pt.vy = -3.639489568414117E7;
			pt.m = mp;
			pt.h = smoothing_length;
			double rho = sin(xi)/xi * rhoc;
			pt.p = r->eos_polytrope.K *rho*rho;
			pt.type = REB_PTYPE_SPH;
			reb_add(r, pt);
		}
		n++;
	}

	double solar_mass = 1.998e33;
	struct reb_particle star = {0};
	star.type = REB_PTYPE_NBODY;
	star.m = solar_mass;
	star.x = 3.123784414852968E5;
	star.y = 1.702036924966430E4;
	star.vx = -1.205460224677914E2;
	star.vy = 1.198395092992375E2;

	reb_add(r, star);

	reb_move_to_com(r);		// This makes sure the planetary systems stays within the computational domain and doesn't drift.
	reb_integrate(r, INFINITY);
}

void heartbeat(struct reb_simulation* r){
	char checkfile[30];
	if (reb_output_check(r, 10.*M_PI)){
		reb_output_ascii(r, "sph.txt");  
		reb_output_timing(r, 0);
		// reb_move_to_com(r);
	}	
	if (reb_output_check(r, 100.)){
		sprintf(checkfile, "checkpoints/checkpoint%04d.h5", (int)round(r->t/100.));
		reb_output_hdf5(r, checkfile);
	}
}