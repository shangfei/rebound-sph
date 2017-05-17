#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "rebound.h"

double e0;

void hb(struct reb_simulation* r){
    double e = reb_tools_energy(r);
    double ep = r->energy_offset;
    struct reb_orbit o = reb_tools_particle_to_orbit(1.,r->particles[1],r->particles[0]);
    printf("t %f    E/E+ %e  %e    N/Nen %d %d   a/e %f %f\n",r->t/2./M_PI,fabs((e-e0)/e0),fabs((ep)/e0),r->N, r->ri_mercurius.encounterN,o.a, o.e);
    if (o.a<20.){
        printf("ae kill\n");
        for (int i=0;i<5;i++){
            printf("%d %f %f %.20f\n", i, r->particles[i].x,r->particles[i].y,r->particles[i].m);
        }
        exit(1);
    }
}


int main(int argc, char* argv[]) {
	struct reb_simulation* r = reb_create_simulation();
    srand(10);
	// Setup callbacks:
	r->integrator = REB_INTEGRATOR_MERCURIUS;
	r->collision = REB_COLLISION_DIRECT;
    r->collision_resolve = reb_collision_resolve_merge;
    r->track_energy_offset = 1;
    r->collision_resolve_keep_sorted = 1;
    //r->usleep = 1000;
    r->heartbeat = hb;
    r->ri_mercurius.rcrit = 3;
    r->ri_mercurius.coordinates = 0;
    r->testparticle_type = 1;
    r->dt = 2.*M_PI* 2.;  // 2 years

    struct reb_particle star = {.m=1.,.r=0.0046524726};
    struct reb_particle planet = reb_tools_orbit2d_to_particle(1.,star,6.9e-06, 25,0,0,0);
    planet.r = 7.8832491e-05; 

    reb_add(r,star);
    reb_add(r,planet);
    r->N_active = r->N;
   
    for (int i=0; i<6000;i++){ 
        double a = reb_random_powerlaw(25.-10.5,25.+10.5,-1);
        double m = r->testparticle_type?6e-6/600.:0;
        //if (i+2==5773) m=0;
        struct reb_particle planetesimal = reb_tools_orbit2d_to_particle(1.,star,m,a,0,0,reb_random_uniform(0.,2.*M_PI));
        planetesimal.r = 9.3466253e-06;
        reb_add(r, planetesimal);
    }

    reb_move_to_com(r);
    e0 = reb_tools_energy(r);
	// Start integration
	reb_integrate(r, INFINITY);

}


