/**
 * @file 	problem.c
 * @brief 	Kepler multi-planetary systems and migration.
 * @author 	Hanno Rein <hanno@hanno-rein.de>
 * @detail 	We study multi-planetary systems discovered 
 * by Kepler and evolve the system with both smooth and
 * stochastic migration forces. i
 * 
 * The prpgram expects the id of the system as a command line 
 * argument (i.e. ./nbody --id=142). See below for detail.
 * 
 * The program can be run in parallel on a standard cluster
 * using the submit.bash PBS script. Create a directory 'out/' 
 * to avoid spamming the main directory with output files. 
 *
 * 
 * @section 	LICENSE
 * Copyright (c) 2011 Hanno Rein
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
#include <string.h>
#include <time.h>
#include "main.h"
#include "tools.h"
#include "input.h"
#include "output.h"
#include "problem.h"
#include "particle.h"
#include "boundaries.h"


// The following arrays have one entry for every particle in the system.
double* tau_a;		// migration timescales
double* tau_e;		// eccentricity damping timescales
double* force_r;	// stochastic force (r-component)
double* force_phi;	// stochastic force (phi-component)


double migration_prefac = 1;	// Used to remove the disc on a smooth timescale (see below)
double period_min = 1000000; 	// Minimum orbital period in the system
double period_max = 0;		// Maximum orbital period in the system

void problem_init(int argc, char* argv[]){
	// This program uses the folowing units (G=1):
	// [mass]   = 1 solar mass
	// [time]   = 1 year / (2*pi)
	// [length] = 1 astronomical unit

	// Setup box (all systems are within 5 astronomical units). 
	boxsize 	= 5;
	init_box();

	// The program expects one command line argument (the system number). 
	int id 	= input_get_int(argc,argv,"id",-1);	
	if (id==-1){
		printf("Need id argument.\n Example: ./nbody --id=143\n\n");
		exit(-1);
	}

	// Read input file and search for system on line number 'id'
	char ch[4096];
	FILE* inputFile = fopen ("input.dat" , "r");
	for (int i=0;i<id;i++) fgets(ch,4095,inputFile);
	fclose(inputFile);
	char* pch = strtok(ch," "); // Name of system
	printf("Simulating system: %s\n",pch);

	// Setup directory
	chdir("out");				// Change to 'out/' directory if it exists.
	char command[4096];
	sprintf(command,"mkdir %s",pch);	// Create output directory with the system name
	system(command);
	chdir(pch);

	// Initial conditions
	// -- add star
	struct particle star;
	star.x  = 0; star.y  = 0; star.z  = 0;
	star.vx = 0; star.vy = 0; star.vz = 0;
	star.ax = 0; star.ay = 0; star.az = 0;
	star.m  = atof(strtok(NULL," "));	// Read stellar mass in units of solar masses
	particles_add(star); 
	
	// -- add planets
	struct particle com = star;		// Initialize planets around the center of mass
	while (pch !=NULL){
		pch = strtok(NULL," ");	if (!pch) continue; 
		double period = atof(pch)*0.017202791;			// Read orbital period in days and convert to codeunits
		//period += period * 0.05 * tools_normal(1.);		// Randomize orbital periods by a small amount	
		pch = strtok(NULL," "); if (!pch) continue;
		double mass   = atof(pch);				// Read planet mass from file (in solar masses)
		if (mass>0 && period>0){
			double a = pow(2.*M_PI/period,-2./3.);		// Semi-major axis in astronomical units
			struct particle p1 = com;		
			p1.m  	= mass;					// Mass in solar masses
			p1.x 	+= a;	
			p1.vy 	+= sqrt( G*(com.m+mass)/a );		// Circular orbit around the center of mass
			p1.vx 	= 0;	p1.vz = 0;
			particles_add(p1); 				// Add particle to the simulation
			
			com = tools_get_center_of_mass(com,p1);
			if (period<period_min) period_min = period;	// Record minimum
			if (period>period_max) period_max = period;	//               /maximum period.
		}
		pch = strtok(NULL," ");	if (!pch) continue; 		// Read stellar mass again (not used)
	}
	
	dt 		= 1.9234567e-3*period_min;			// Timestep is a small fraction of innermost orbital period

	force_r 	= calloc(N,sizeof(double));			// Arrays for stochastic forces
	force_phi 	= calloc(N,sizeof(double));

	tau_a = calloc(N,sizeof(double));				// Arrays for smooth migration forces
	tau_e = calloc(N,sizeof(double));

	tau_a[N-1] = 2.*M_PI*1e4;  					// Set the migration timescale of the outermost planet to 1e4 years	
	tau_e[N-1] = 0.1*tau_a[N-1];					// Set the eccentricity damping timescale of that planet to 1e3 years

	tmax = period_max*1e4;						// Integrate for 1e4 outer planer orbits

	tools_move_to_center_of_momentum();				
}


// This routine adds semi-major axis damping.
// See Lee & Peale (2002) for a detailed description. 
void problem_adot(){
	struct particle com = particles[0];
	for(int i=1;i<N;i++){
		if (tau_a[i]!=0){
			struct particle* p = &(particles[i]);
			double tmpfac = migration_prefac*dt/(tau_a[i]);
			// position
			p->x  -= (p->x-com.x)*tmpfac;
			p->y  -= (p->y-com.y)*tmpfac;
			p->z  -= (p->z-com.z)*tmpfac;
			// velocity
			p->vx  += 0.5 * (p->vx-com.vx)*tmpfac;
			p->vy  += 0.5 * (p->vx-com.vy)*tmpfac;
			p->vz  += 0.5 * (p->vx-com.vz)*tmpfac;
		}
		com =tools_get_center_of_mass(com,particles[i]);
	}
}

// This routine adds eccentricity damping.
// This one is more complicated than the migration routine because it needs orbital elements to compute the forces.
void problem_edot(){
	struct particle com = particles[0];
	for(int i=1;i<N;i++){
		if (tau_e[i]!=0 ){
			struct particle* p = &(particles[i]);
			struct orbit o = tools_p2orbit(*p,com);
			if (o.e>1e-8){
			double d = migration_prefac*dt/(tau_e[i]);
			double rdot  = o.h/o.a/( 1. - o.e*o.e ) * o.e * sin(o.f);
			double rfdote = o.h/o.a/( 1. - o.e*o.e ) * ( 1. + o.e*cos(o.f) ) * (o.e + cos(o.f)) / (1.-o.e*o.e) / (1.+o.e*cos(o.f));
			//position
			double tmpfac = d * (  o.r/(o.a*(1.-o.e*o.e)) - (1.+o.e*o.e)/(1.-o.e*o.e));
			p->x -= tmpfac * (p->x-com.x);
			p->y -= tmpfac * (p->y-com.y);
			p->z -= tmpfac * (p->z-com.z);
			//vx
			tmpfac = rdot/(o.e*(1.-o.e*o.e));
			p->vx -= d * o.e * (   cos(o.Omega) *      (tmpfac * cos(o.omega+o.f) - rfdote*sin(o.omega+o.f) )
						-cos(o.inc) * sin(o.Omega) * (tmpfac * sin(o.omega+o.f) + rfdote*cos(o.omega+o.f) ));
			//vy
			p->vy -= d * o.e * (   sin(o.Omega) *      (tmpfac * cos(o.omega+o.f) - rfdote*sin(o.omega+o.f) )
						+cos(o.inc) * cos(o.Omega) * (tmpfac * sin(o.omega+o.f) + rfdote*cos(o.omega+o.f) ));
			//vz
			p->vz -= d * o.e * (     sin(o.inc) *      (tmpfac * sin(o.omega+o.f) + rfdote*cos(o.omega+o.f) ));
			}
		
		}
		com =tools_get_center_of_mass(com,particles[i]);
	}
}	
	

// This routine adds stochastic kicks which have an exponentially decaying autocorrelation timescale.
//  
// References: - Kasdin, N.J. 1995, Proceedings of the IEEE, Vol 83, NO. 5.
//             - Rein 2011 (Thesis). 
void problem_kicks(){
	double D=.7e-6;					// Strength of stochastic kicks relative to gravity
	double tau = period_max/2.;			// Autocorrelation timescale
	struct particle com = particles[0];
	for(int i=1;i<N;i++){
		struct particle* p = &(particles[i]);
		struct orbit o = tools_p2orbit(*p,com);
		force_r[i] *= exp(-dt/tau);		// Evolve r and phi components individually
		force_r[i] += tools_normal(1.-exp(-2.0*dt/tau));
		force_phi[i] *= exp(-dt/tau);
		force_phi[i] += tools_normal(1.-exp(-2.0*dt/tau));
		double dx = particles[i].x - com.x;
		double dy = particles[i].y - com.y;
		double dz = particles[i].z - com.z;
		double r = sqrt(dx*dx + dy*dy + dz*dz);
		double prefact = -D*G/(r*r*r)*com.m;
		particles[i].vx += dt*prefact*sin(o.f)*force_r[i]; 
		particles[i].vy += dt*prefact*cos(o.f)*force_r[i]; 
		particles[i].vx += dt*prefact*cos(o.f)*force_phi[i]; 
		particles[i].vy += dt*prefact*sin(o.f)*force_phi[i]; 
		com =tools_get_center_of_mass(com,particles[i]);
	}
}

void problem_inloop(){
}

void output_period_ratio(char* filename){
	FILE* of = fopen(filename,"a+"); 
	struct particle com = particles[0];
	struct orbit o1 = tools_p2orbit(particles[1],com);
	com = tools_get_center_of_mass(com,particles[1]);
	for(int i=2;i<N;i++){
		struct orbit o2 = tools_p2orbit(particles[i],com);
		fprintf(of,"%e\t%e\n",t,o2.P/o1.P);
		com =tools_get_center_of_mass(com,particles[i]);
		o1 = o2;
	}
	fclose(of);
}

void problem_output(){
	if (N<3) exit_simulation = 1;			// Exit simulation if there are 2 or less particles in the system
	if(output_check(1000.*period_max)){
		tools_move_to_center_of_momentum();	// Periodically move to the center of mass to avoid mean drift
	}
	if(output_check(5.436542264*period_max)){	// Output data periodically
		output_append_orbits("orbits.txt");
		output_period_ratio("period_ratio.txt");
	}
	
	// The following lines can be used to smoothly remove the migration forces.	
	//double mig_t1 = period_max*5000;
	//double mig_t2 = period_max*5500;
	//if (t>mig_t1){
	//	if (t<mig_t2){
	//		migration_prefac = 1.-  (t-mig_t1)/(mig_t2-mig_t1);
	//	}else{
	//		migration_prefac = 0;
	//	}
	//}

	if (migration_prefac>0){
		problem_adot();				// Add migration force
		problem_edot();				// Add exccentricity damping force
	}
	problem_kicks();				// Add stochastic migration
}

void problem_finish(){
}
