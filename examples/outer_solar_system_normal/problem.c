/**
 * Outer Solar System
 *
 * This example uses the IAS15 integrator
 * to integrate the outer planets of the solar system. The initial 
 * conditions are taken from Applegate et al 1986. Pluto is a test
 * particle. This example is a good starting point for any long term orbit
 * integrations.
 *
 * You probably want to turn off the visualization for any serious runs.
 * Go to the makefile and set `OPENGL=0`. 
 *
 * The example also works with the WHFAST symplectic integrator. We turn
 * off safe-mode to allow fast and accurate simulations with the symplectic
 * corrector. If an output is required, you need to call ireb_integrator_synchronize()
 * before accessing the particle structure.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "rebound.h"

double ss_pos[6][3] =
    {
     {-4.06428567034226e-3, -6.08813756435987e-3, -1.66162304225834e-6}, // Sun
     {+3.40546614227466e+0, +3.62978190075864e+0, +3.42386261766577e-2}, // Jupiter
     {+6.60801554403466e+0, +6.38084674585064e+0, -1.36145963724542e-1}, // Saturn
     {+1.11636331405597e+1, +1.60373479057256e+1, +3.61783279369958e-1}, // Uranus
     {-3.01777243405203e+1, +1.91155314998064e+0, -1.53887595621042e-1}, // Neptune
     {-2.13858977531573e+1, +3.20719104739886e+1, +2.49245689556096e+0}  // Pluto
};
double ss_vel[6][3] =
    {
     {+6.69048890636161e-6, -6.33922479583593e-6, -3.13202145590767e-9}, // Sun
     {-5.59797969310664e-3, +5.51815399480116e-3, -2.66711392865591e-6}, // Jupiter
     {-4.17354020307064e-3, +3.99723751748116e-3, +1.67206320571441e-5}, // Saturn
     {-3.25884806151064e-3, +2.06438412905916e-3, -2.17699042180559e-5}, // Uranus
     {-2.17471785045538e-4, -3.11361111025884e-3, +3.58344705491441e-5}, // Neptune
     {-1.76936577252484e-3, -2.06720938381724e-3, +6.58091931493844e-4}  // Pluto
};

double ss_mass[6] =
    {
     1.00000597682, // Sun + inner planets
     1. / 1047.355, // Jupiter
     1. / 3501.6,   // Saturn
     1. / 22869.,   // Uranus
     1. / 19314.,   // Neptune
     7.4074074e-09  // Pluto
};

double tmax = 365*1000;


int main(int argc, char* argv[]) {
	struct reb_simulation* r = reb_create_simulation();
	// Setup constants
	const double k = 0.01720209895; // Gaussian constant
	r->dt = 11.4;			// in days
	r->G = k * k;			// These are the same units as used by the mercury6 code.

	// Setup callbacks:
	r->integrator = REB_INTEGRATOR_MERCURIUS;
	//r->integrator = REB_INTEGRATOR_WHFASTHELIO;
	//r->integrator	= REB_INTEGRATOR_IAS15;
	r->gravity	= REB_GRAVITY_MERCURIUS;
    r->ri_mercurius.rcrit = 6.*5.2*powf(50.*1e-3/3,1./3.);
    r->ri_mercurius.coordinates = 0;
   //  r->dt *=0.00915;
    //r->softening = 1e-2*r->ri_mercurius.rcrit;
    r->usleep = 1000;

	// Initial conditions
	for (int i = 0; i < 6; i++) {
		struct reb_particle p = {0};
		p.x = ss_pos[i][0];
		p.y = ss_pos[i][1];
		p.z = ss_pos[i][2];
		p.vx = ss_vel[i][0];
		p.vy = ss_vel[i][1];
		p.vz = ss_vel[i][2];
		p.m = ss_mass[i];
        if (i>0){
            p.m*=50.;
        }
		reb_add(r, p);
	}

    reb_move_to_com(r);


	// Start integration
	reb_integrate(r, tmax);

}
