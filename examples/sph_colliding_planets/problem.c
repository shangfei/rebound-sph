/**
 * Colliding planets
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include "rebound.h"

void heartbeat(struct reb_simulation* r);

int main(int argc, char* argv[]){
	struct reb_simulation* r = reb_create_simulation();
	r->gravity	= REB_GRAVITY_TREE;
	r->boundary	= REB_BOUNDARY_OPEN;
	r->opening_angle2	= 1.e-2;		// This constant determines the accuracy of the tree code gravity estimate.
	r->G 		= 6.674e-8;		
	r->softening 	= 0.02;		// Gravitational softening length
	r->dt 		= 0.05;		// Timestep
	r->initSPH	= 1;
	const double boxsize = 1e11;
	r->integrator 	= REB_INTEGRATOR_LEAPFROG;
	r->heartbeat  	= heartbeat;
	r->eos				= REB_EOS_POLYTROPE;
	r->eos_polytrope.n 	= 1;
	r->eos_polytrope.K	= 2.6e12; 
	
	r->hydro.nnmin = 45;
	r->hydro.nnmax = 55;

	reb_configure_box(r,boxsize,1,1,1);
	
	double mjupiter = 1.898e30;// 4*M_PI*M_PI;
	int N = 2500;
	const double K = 2.6e12;
	double alpha = sqrt(K/2./M_PI/r->G);
	int Nbin = 50;
	double dxi = M_PI/( (double) Nbin);
	double mp = mjupiter / (double)N;
	r->m = mp;
	double R = 7.9e9;
	double smoothing_length = R/ 5.; 
	double rhoc = 5.;
	int n = 0;

	for (int i=0;i<Nbin;i++){
		if (n>=N) break;
		struct reb_particle pt = {0};
		double xi1 = dxi*i;
		double xi2 = dxi * (i+1);
		int N_in_bin =  (int)round((sin(xi2)-xi2*cos(xi2)-sin(xi1)+xi1*cos(xi1))/M_PI*(double)N);
		for (int j=0;j<N_in_bin;j++){
			double phi 	= reb_random_uniform(0,2.*M_PI);
			double cos_theta = reb_random_uniform(-1, 1);
			double xi = reb_random_uniform(xi1, xi2);
			pt.x = alpha * xi * sin(acos(cos_theta))*cos(phi) - 1.5*R;
			pt.y = alpha * xi * sin(acos(cos_theta))*sin(phi) -0.5*R;
			pt.z = alpha * xi * cos_theta;
			pt.vx =  0.5*sqrt(2.*r->G*mjupiter/3./R);
			pt.m = mp;
			pt.rho = sin(xi)/xi * rhoc;
			pt.h = smoothing_length;
			pt.p = K*pt.rho*pt.rho;
			pt.type = REB_PTYPE_SPH;			
			reb_add(r, pt);
		}
		n++;
	}

	for (int i=0;i<Nbin;i++){
		if (n>=N) break;
		struct reb_particle pt = {0};
		double xi1 = dxi*i;
		double xi2 = dxi * (i+1);
		int N_in_bin =  (int)round((sin(xi2)-xi2*cos(xi2)-sin(xi1)+xi1*cos(xi1))/M_PI*(double)N);
		for (int j=0;j<N_in_bin;j++){
			double phi 	= reb_random_uniform(0,2.*M_PI);
			double cos_theta = reb_random_uniform(-1, 1);
			double xi = reb_random_uniform(xi1, xi2);
			pt.x = alpha * xi * sin(acos(cos_theta))*cos(phi) + 1.5*R;
			pt.y = alpha * xi * sin(acos(cos_theta))*sin(phi) + 0.5*R;
			pt.z = alpha * xi * cos_theta;
			pt.vx = -0.5*sqrt(2.*r->G*mjupiter/3./R);
			pt.m = mp;
			pt.rho = sin(xi)/xi * rhoc;
			pt.h = smoothing_length;
			reb_add(r, pt);
		}
		n++;
	}

	/* To restart a simulation, comment out the lines above and uncomment the following snippet. Don't forget to change the name of the binary file to restart the simulation. 
	struct reb_simulation* r = reb_create_simulation_from_binary("restart0009.dat");
	r->heartbeat = heartbeat;
	*/

	reb_move_to_com(r);		// This makes sure the planetary systems stays within the computational domain and doesn't drift.
	reb_integrate(r, INFINITY);
}

void heartbeat(struct reb_simulation* r){
	char checkfile[30];
	if (reb_output_check(r, 50.)){  
		reb_output_timing(r, 0);
		reb_output_ascii(r, "sph.txt");
		// reb_move_to_com(r);
	}
	if (reb_output_check(r, 250.)){
		sprintf(checkfile, "checkpoint%04d.h5", (int)round(r->t/250.));
		reb_output_hdf5(r, checkfile);
	}
}
