#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "rebound.h"

void gr_force(struct reb_simulation* const r){
    // From REBOUNDx
    const struct reb_particle source = r->particles[0];
    const double C2 = 1.0130251e+08;
    const double prefac1 = 6.*(r->G*source.m)*(r->G*source.m)/C2;
    
    for (int i=1;i<r->N;i++){
        const struct reb_particle p = r->particles[i];
        const double dx = p.x - source.x;
        const double dy = p.y - source.y;
        const double dz = p.z - source.z;
        const double r2 = dx*dx + dy*dy + dz*dz;
        const double prefac = prefac1/(r2*r2);
        r->particles[i].ax -= prefac*dx;
        r->particles[i].ay -= prefac*dy;
        r->particles[i].az -= prefac*dz;
        r->particles[0].ax += p.m/source.m*prefac*dx;
        r->particles[0].ay += p.m/source.m*prefac*dy;
        r->particles[0].az += p.m/source.m*prefac*dz;
    }
}

int main(int argc, char* argv[]) {
    // Restart
    struct reb_simulation* r = reb_simulationarchive_restart("restart_0051.bin");
    reb_simulationarchive_load_snapshot(r,"restart_0051.bin",159047);

    // Randomize
    srand(getpid() * time(NULL));
    double shift =0.5-((double)rand()/(double)RAND_MAX);
    r->ri_whfast.p_j[3].x += 1e-10 *shift;

    // New SA
    r->simulationarchive_filename = "restart_0051_col.bin";
    r->simulationarchive_walltime = 0.;
    r->simulationarchive_next = 0.;
    r->simulationarchive_interval_walltime = 0.;

    // Run sim
    r->additional_forces = gr_force;
    reb_integrate(r, 5.1553e10); //10 Gyr
}

