/**
 * @file 	problem.c
 * @brief 	Example problem: forced migration of GJ876.
 * @author 	Hanno Rein <hanno@hanno-rein.de>
 * @detail 	This example applies dissipative forces to two
 * bodies orbiting a central object. The forces are specified
 * in terms of damping timescales for the semi-major axis and
 * eccentricity. This mimics planetary micration in a proto-
 * stellar disc. The example reproduces the study of Lee & 
 * Peale (2002) on the formation of the planetary system 
 * GJ876. For a comparison, see figure 4 in their paper.
 *
 * 
 * @section 	LICENSE
 * Copyright (c) 2011 Hanno Rein, Shangfei Liu
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


double* tau_a;
double* tau_e;
double period_min = 1000000;
double period_max = 0;

void problem_init(int argc, char* argv[]){
	int id 	= input_get_int(argc,argv,"id",-1);	// Inital timestep in days
	if (id==-1){
		printf("Need id argument.\n");
		exit(-1);
	}
	id-=1;

	char ch[4096];
	FILE* inputFile = fopen ("input.dat" , "r");
	for (int i=0;i<=id;i++) fgets(ch,4095,inputFile);
	fclose(inputFile);
	char* pch = strtok(ch," "); // Name
	printf("Simulating system: %s\n",pch);
	chdir("out");
	char command[4096];
	sprintf(command,"mkdir %s",pch);
	system(command);
	chdir(pch);

	// Setup constants
	boxsize 	= 5;
	init_box();

	// Initial conditions
	struct particle star;
	star.x  = 0; star.y  = 0; star.z  = 0;
	star.vx = 0; star.vy = 0; star.vz = 0;
	star.ax = 0; star.ay = 0; star.az = 0;
	star.m  = atof(strtok(NULL," "));	// in solar masses
	particles_add(star); 
	
	struct particle com = star;
	double period_last = 0;
	while (pch !=NULL){
		pch = strtok(NULL," ");	if (!pch) continue; 
		double period = atof(pch)*0.017202791;// from days to codeunits
		
		//period += period * 0.05 * tools_normal(1.);	
		if (N>1){
		//	period=period_last*2.7;
		//	period+= period*0.2*tools_normal(1);
		}
		period_last = period;

		pch = strtok(NULL," "); if (!pch) continue;
		double mass   = atof(pch);	// in solar masses

		if (mass>0 && period>0){
			double a = pow(2.*M_PI/period,-2./3.);
			struct particle p1 = com;
			p1.m  	= mass;
			p1.x 	+= a;	
			p1.vy 	+= sqrt( G*(com.m+mass)/a );
			p1.vx 	= 0;	p1.vz = 0;
			particles_add(p1); 
			
			com = tools_get_center_of_mass(com,p1);
			if (period<period_min) period_min = period;
			if (period>period_max) period_max = period;
		}
		pch = strtok(NULL," ");	if (!pch) continue; 	// Stellar mass - ignore
	}
	
	dt 		= 1.9234567e-3*period_min;


	tau_a = calloc(N,sizeof(double));
	tau_e = calloc(N,sizeof(double));

	tau_a[N-1] = 2.*M_PI*1e3;  // 1e4 years
	tau_e[N-1] = 0.1*tau_a[N-1];

	for (int i=1;i<N-1;i++){
		//tau_e[i] = 1e12;
	}

	tmax = period_max*1e4;

	tools_move_to_center_of_momentum();
}

double migration_prefac = 1;

// Semi-major axis damping
void problem_adot(){
	struct particle com = particles[0];
	for(int i=1;i<N;i++){
		if (tau_a[i]!=0){
			struct particle* p = &(particles[i]);
			//struct orbit o = tools_p2orbit(*p,com);
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

// Eccentricity damping
// This one is more complicated as it needs orbital elements to compute the forces.
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
	

void problem_kicks(){
	double D = 5e-5;
	struct particle star = particles[0];
	srand(floor(t/(period_min*0.24234234)));
	for(int i=N-1;i<N;i++){
		double dx = particles[i].x - star.x;
		double dy = particles[i].y - star.y;
		double dz = particles[i].z - star.z;
		double r = sqrt(dx*dx + dy*dy + dz*dz);
		double prefact = -G/(r*r*r)*star.m;
		particles[i].vx += dt*prefact*dx*tools_normal(1.)*D; 
		particles[i].vy += dt*prefact*dy*tools_normal(1.)*D; 
		particles[i].vz += dt*prefact*dz*tools_normal(1.)*D; 
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
	if (N<3) exit_simulation = 1;
	if(output_check(1000.*period_max)){
		tools_move_to_center_of_momentum();
	}
	//if(output_check(10000.*dt)){
	//	output_timing();
	//}
	if(output_check(5.436542264*period_max)){
		output_append_orbits("orbits.txt");
		output_period_ratio("period_ratio.txt");
	}
	
	
	
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
		problem_adot();
		problem_edot();
	}
	//problem_kicks();
}

void problem_finish(){
}
