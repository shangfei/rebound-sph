#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "rebound.h"

int main(int argc, char* argv[]) {
	struct reb_simulation* r = reb_create_simulation_from_binary("outerss.bin");
	r->dt = 11.4/365.25*2.*M_PI;			// in days
    r->visualization = REB_VISUALIZATION_OPENGL;

	r->integrator = REB_INTEGRATOR_MERCURIUS;
	//r->integrator = REB_INTEGRATOR_WHFASTHELIO;
	//r->integrator	= REB_INTEGRATOR_IAS15;
    r->ri_mercurius.rcrit = 3.;
    r->ri_mercurius.coordinates = 0;
    r->usleep = 1000;

    reb_move_to_com(r);

    double tmax = 2.*M_PI*2000;
	reb_integrate(r, tmax);
}
