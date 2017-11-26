/**
 * @file 	hydro.c
 * @brief 	Smoothed particle hydrodynamics
 * @author 	Shangfei Liu <shangfei.liu@gmail.com>
 *
 * @details
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include "particle.h"
#include "rebound.h"
#include "tree.h"
#include "boundary.h"
#include "eos.h"
#define MAX(a, b) ((a) > (b) ? (a) : (b))    ///< Returns the maximum of a and b
#define MIN(a, b) ((a) > (b) ? (b) : (a))	 ///< Returns the minimum of a and b

#ifdef MPI
#include "communication_mpi.h"
#endif

/**
  * @brief The function loops over all trees to call calculate_forces_for_particle_from_cell() tree to calculate forces for each particle.
  * @param r REBOUND simulation to consider
  * @param pt Index of the particle the force is calculated for.
  * @param gb Ghostbox plus position of the particle (precalculated). 
  */
static void reb_calculate_acceleration_for_nbody_particle(const struct reb_simulation* const r, const int pt, const struct reb_ghostbox gb);

static void reb_calculate_acceleration_for_sph_particle(const struct reb_simulation* const r, const int pt, const struct reb_ghostbox gb);

static void reb_calculate_acceleration_for_nongravitating_sph_particle(const struct reb_simulation* const r, const int pt, const struct reb_ghostbox gb);

static double kernel_center(double h);

void reb_init_hydrodynamics(struct reb_simulation* r){
	struct reb_particle* const particles = r->particles;
	const int N = r->N;
	// const int N_active = r->N_active;
	// const double G = r->G;
	switch (r->eos) {
		case REB_EOS_DUMMY:
			break;
		case REB_EOS_POLYTROPE:
			r->hydro.gamma = r->eos_polytrope.gamma;
			break;
		case REB_EOS_GAMMA_LAW:
			r->hydro.gamma = r->eos_gammalaw.gamma;
			break;
		case REB_EOS_ISOTHERMAL:
			break;
	}

	// const double softening2 = r->softening*r->softening;
	// const unsigned int _gravity_ignore_terms = r->gravity_ignore_terms;
	// const int _N_active = ((N_active==-1)?N:N_active) - r->N_var;
	// const int _N_real   = N  - r->N_var;
	// const int _testparticle_type   = r->testparticle_type;
// #pragma omp parallel for schedule(guided)
	for (int i=0; i<N; i++){
		particles[i].ax 	= 0.; 
		particles[i].ay 	= 0.; 
		particles[i].az 	= 0.;
		if (particles[i].type == REB_PTYPE_SPH) {
			particles[i].rho 	= 0.;
			particles[i].rhoi 	= 0.;
			particles[i].nn 	= 0;
		// Make sure the pressure is not set to zero.
			particles[i].e 		= 0;
		}
	}

	int init_step = 0, nnmin = 0, nnmax = 0;
	char filename[30] = "smoothinglength.txt";
	FILE* of = fopen(filename, "a");
	while(r->initSPH){
		int need_iteration =0;
		init_step += 1;
		fprintf(of, "Number of iterations of hydro initialization: %i, the smallest and largest nn values are %i %i \n", init_step, nnmin, nnmax);

	// Summing over all Ghost Boxes
		for (int gbx=-r->nghostx; gbx<=r->nghostx; gbx++){
		for (int gby=-r->nghosty; gby<=r->nghosty; gby++){
		for (int gbz=-r->nghostz; gbz<=r->nghostz; gbz++){
		// Summing over all particle pairs
// #pragma omp parallel for schedule(guided)
			for (int i=0; i<N; i++){
#ifndef OPENMP
				if (reb_sigint) return;
#endif // OPENMP
				struct reb_ghostbox gb = reb_boundary_get_ghostbox(r, gbx,gby,gbz);
				// Precalculated shifted position
				gb.shiftx += particles[i].x;
				gb.shifty += particles[i].y;
				gb.shiftz += particles[i].z;
				// Get the number of neighboring particles to update the smoothing length
				if (particles[i].type == REB_PTYPE_SPH) {
					particles[i].nn = 0;
					particles[i].rho = 0.;
					reb_calculate_acceleration_for_sph_particle(r, i, gb);
					particles[i].rhoi = particles[i].m * kernel_center(particles[i].h);		
					particles[i].rho += particles[i].rhoi; // Density contribution from the particle itself.
					if (r->eos != REB_EOS_DUMMY) reb_calculate_internal_energy_for_sph_particle(r, i); // Initialize the internal energy of the particle
					reb_eos(r, i);			
					nnmin = MIN(nnmin, particles[i].nn);
					nnmax = MAX(nnmax, particles[i].nn);
					if (particles[i].nn > r->hydro.nnmax || particles[i].nn < r->hydro.nnmin) {
						need_iteration = 1;
						fprintf(of, "Particle %i: nn = %i \t old h = %e \t", i, particles[i].nn, particles[i].h);
						if (particles[i].nn != 0) {
							particles[i].h *= 0.5*(1+cbrt(50./((double)particles[i].nn)));
						} else if (particles[i].nn > 55) {
							particles[i].h /= 2.0;
						} else if (particles[i].nn < 45) {
							particles[i].h *= 2.0;
						}
						
						fprintf(of, "new h = %e pos: %e %e %e \n", particles[i].h, particles[i].x, particles[i].y, particles[i].z);
					
					}
				} else {
					fprintf(of, "nbody particles %i pos: %e %e %e \n", i, particles[i].x, particles[i].y, particles[i].z);
					reb_calculate_acceleration_for_nbody_particle(r, i, gb);										
				}
				// particles[i].rho += particles[i].m * kernel_center(particles[i].h);		// Density contribution from the particle itself.
				// // Calculate the density and number of neighbors again using the updated smoothing length
				// particles[i].rho = 0.;
				// particles[i].nn = 0;
				particles[i].ax = 0; 
				particles[i].ay = 0; 
				particles[i].az = 0;
				// reb_calculate_gravitational_acceleration_for_sph_particle(r, i, gb);
				// particles[i].rho += particles[i].m * kernel_center(particles[i].h);						
			
			}
			
		}
		}
		}
		if (need_iteration == 0) {
			for (int gbx=-r->nghostx; gbx<=r->nghostx; gbx++){
			for (int gby=-r->nghosty; gby<=r->nghosty; gby++){
			for (int gbz=-r->nghostz; gbz<=r->nghostz; gbz++){
			for (int i=0;i<N;i++) {
				struct reb_ghostbox gb = reb_boundary_get_ghostbox(r, gbx,gby,gbz);
				// Precalculated shifted position
				gb.shiftx += particles[i].x;
				gb.shifty += particles[i].y;
				gb.shiftz += particles[i].z;
				// Get the number of neighboring particles to update the smoothing length
				if (particles[i].type == REB_PTYPE_SPH) {
					particles[i].rho = 0.;
					particles[i].nn = 0;					
					reb_calculate_acceleration_for_sph_particle(r, i, gb);					
					particles[i].rhoi = particles[i].m * kernel_center(particles[i].h);		
					particles[i].rho += particles[i].rhoi; // Density contribution from the particle itself.
					if (r->eos == REB_EOS_ISOTHERMAL) {
						double dx = gb.shiftx - particles[0].x;
						double dy = gb.shifty - particles[0].y;
						double dz = gb.shiftz - particles[0].z;
						double _r = sqrt(dx*dx + dy*dy + dz*dz);
						particles[i].cs = r->eos_isothermal.cs0 * pow(_r, r->eos_isothermal.q);
					} else {
						particles[i].cs = sqrt(r->hydro.gamma * particles[i].p / particles[i].rhoi);
					}
					if (r->eos != REB_EOS_DUMMY) reb_calculate_internal_energy_for_sph_particle(r, i); // Initialize the internal energy of the particle
					reb_eos(r, i);
				} else {
					reb_calculate_acceleration_for_nbody_particle(r, i, gb);					
				}
				particles[i].ax = 0.; 
				particles[i].ay = 0.; 
				particles[i].az = 0.;
			}
			}
			}
			}
			r->initSPH = 0;
		}	
	}
	fclose(of);
}


void reb_evolve_hydrodynamics(struct reb_simulation* r){
	struct reb_particle* const particles = r->particles;
	const int N = r->N;
	const int N_active = r->N_active;
	const double G = r->G;
	const double softening2 = r->softening*r->softening;
	// const unsigned int _gravity_ignore_terms = r->gravity_ignore_terms;
	const int _N_active = ((N_active==-1)?N:N_active) - r->N_var;
	const int _N_real   = N  - r->N_var;
	const int _testparticle_type   = r->testparticle_type;
	switch (r->gravity){
		case REB_GRAVITY_NONE: // Do nothing.
		break;

		case REB_GRAVITY_BASIC:
		{
			const int nghostx = r->nghostx;
			const int nghosty = r->nghosty;
			const int nghostz = r->nghostz;
#pragma omp parallel for 
			for (int i=0; i<N; i++){
				particles[i].ax = 0; 
				particles[i].ay = 0; 
				particles[i].az = 0; 
			}
			// Summing over all Ghost Boxes
			for (int gbx=-nghostx; gbx<=nghostx; gbx++){
			for (int gby=-nghosty; gby<=nghosty; gby++){
			for (int gbz=-nghostz; gbz<=nghostz; gbz++){
				struct reb_ghostbox gb = reb_boundary_get_ghostbox(r, gbx,gby,gbz);
				// Summing over all particle pairs
#pragma omp parallel for
				for (int i=0; i<_N_real; i++){
#ifndef OPENMP
                if (reb_sigint) return;
#endif // OPENMP
				for (int j=0; j<_N_active; j++){
					// if (_gravity_ignore_terms==1 && ((j==1 && i==0) || (i==1 && j==0) )) continue;
					// if (_gravity_ignore_terms==2 && ((j==0 || i==0) )) continue;
					if (i==j) continue;
					const double dx = (gb.shiftx+particles[i].x) - particles[j].x;
					const double dy = (gb.shifty+particles[i].y) - particles[j].y;
					const double dz = (gb.shiftz+particles[i].z) - particles[j].z;
					const double _r = sqrt(dx*dx + dy*dy + dz*dz + softening2);
					const double prefact = -G/(_r*_r*_r)*particles[j].m;
					
					particles[i].ax    += prefact*dx;
					particles[i].ay    += prefact*dy;
					particles[i].az    += prefact*dz;
				}
				}
                if (_testparticle_type){
				for (int i=0; i<_N_active; i++){
#ifndef OPENMP
                if (reb_sigint) return;
#endif // OPENMP
				for (int j=_N_active; j<_N_real; j++){
					// if (_gravity_ignore_terms==1 && ((j==1 && i==0) )) continue;
					// if (_gravity_ignore_terms==2 && ((j==0 || i==0) )) continue;
					const double dx = (gb.shiftx+particles[i].x) - particles[j].x;
					const double dy = (gb.shifty+particles[i].y) - particles[j].y;
					const double dz = (gb.shiftz+particles[i].z) - particles[j].z;
					const double _r = sqrt(dx*dx + dy*dy + dz*dz + softening2);
					const double prefact = -G/(_r*_r*_r)*particles[j].m;
					
					particles[i].ax    += prefact*dx;
					particles[i].ay    += prefact*dy;
					particles[i].az    += prefact*dz;
				}
				}
                }
			}
			}
			}
		}
		break;

		case REB_GRAVITY_COMPENSATED:
		break;

		case REB_GRAVITY_TREE:
		{
#pragma omp parallel for schedule(guided)
			for (int i=0; i<N; i++){
				particles[i].ax = 0; 
				particles[i].ay = 0; 
				particles[i].az = 0;
				particles[i].oldrho = particles[i].rho;
				particles[i].rho = 0;
				particles[i].nn = 0;
				particles[i].oldp = particles[i].p;
			}
			const double tau = 0.3; // Need to move it to other place.
			double newdt = r->dt;
			// Summing over all Ghost Boxes
			for (int gbx=-r->nghostx; gbx<=r->nghostx; gbx++){
			for (int gby=-r->nghosty; gby<=r->nghosty; gby++){
			for (int gbz=-r->nghostz; gbz<=r->nghostz; gbz++){
				// double cs = 0;
				// Summing over all particle pairs
#pragma omp parallel for schedule(guided)
				for (int i=0; i<N; i++){
#ifndef OPENMP
                    if (reb_sigint) return;
#endif // OPENMP
					struct reb_ghostbox gb = reb_boundary_get_ghostbox(r, gbx,gby,gbz);
					// Precalculated shifted position
					gb.shiftx += particles[i].x;
					gb.shifty += particles[i].y;
					gb.shiftz += particles[i].z;
					if (particles[i].type == REB_PTYPE_SPH) {						
						particles[i].rhoi = particles[i].m * kernel_center(particles[i].h); 	// Density contribution from the particle itself.
						reb_eos(r, i);
						reb_calculate_acceleration_for_sph_particle(r, i, gb);
						particles[i].h *= 0.5*(1+cbrt(50./((double)particles[i].nn)));
						if (particles[i].h <= 0) particles[i].h = r->boxsize_max/2.;
						if (particles[i].h > r->boxsize_max) particles[i].h = r->boxsize_max/2.;	
						particles[i].rho += particles[i].rhoi;
						if (r->eos == REB_EOS_ISOTHERMAL) {
							double _r;
							if (r->N_active>1) {
								double dx1 = gb.shiftx - particles[0].x;
								double dy1 = gb.shifty - particles[0].y;
								double dz1 = gb.shiftz - particles[0].z;
								double dx2 = gb.shiftx - particles[1].x;
								double dy2 = gb.shifty - particles[1].y;
								double dz2 = gb.shiftz - particles[1].z;
								_r = MIN(dx1*dx1 + dy1*dy1 + dz1*dz1, dx1*dx1 + dy1*dy1 + dz1*dz1);
								_r = sqrt(_r);
							} else {
								double dx = gb.shiftx - particles[0].x;
								double dy = gb.shifty - particles[0].y;
								double dz = gb.shiftz - particles[0].z;
								_r = sqrt(dx*dx + dy*dy + dz*dz);
							}
							particles[i].cs = r->eos_isothermal.cs0 * pow(_r, r->eos_isothermal.q);
						} else {
							particles[i].cs = sqrt(r->hydro.gamma * particles[i].p / particles[i].rhoi);
						}
						newdt = MIN(newdt, tau*particles[i].h/particles[i].cs);
					} else {
						reb_calculate_acceleration_for_nbody_particle(r, i, gb);
					}				
				}
				
			}
			}
			}
			//Todo: make the dt more adjustable.
			r->dt = newdt; // If the number of sph particles are not many and when they are collaping, the dt may becomes -inf and N_tot is 0 (I don't know why it becomes zero.)
		}
		break;

		case REB_GRAVITY_TREE_TESTPARTICLE:
		{
#pragma omp parallel for schedule(guided)
			for (int i=0; i<N; i++){
				particles[i].ax = 0; 
				particles[i].ay = 0; 
				particles[i].az = 0;
				particles[i].oldrho = particles[i].rho;
				particles[i].rho = 0;
				particles[i].nn = 0;
				particles[i].oldp = particles[i].p;
			}
			const double tau = 0.3; // Need to move it to other place.
			double newdt = r->dt;
			// Summing over all Ghost Boxes
			for (int gbx=-r->nghostx; gbx<=r->nghostx; gbx++){
			for (int gby=-r->nghosty; gby<=r->nghosty; gby++){
			for (int gbz=-r->nghostz; gbz<=r->nghostz; gbz++){
				// double cs = 0;
				// Summing over all particle pairs
#pragma omp parallel for schedule(guided)
				for (int i=0; i<N; i++){
#ifndef OPENMP
                    if (reb_sigint) return;
#endif // OPENMP
					if (particles[i].type == REB_PTYPE_SPH) {
						struct reb_ghostbox gb = reb_boundary_get_ghostbox(r, gbx,gby,gbz);
						// Precalculated shifted position
						gb.shiftx += particles[i].x;
						gb.shifty += particles[i].y;
						gb.shiftz += particles[i].z;
						particles[i].rhoi = particles[i].m * kernel_center(particles[i].h); 	// Density contribution from the particle itself.
						reb_eos(r, i);
						reb_calculate_acceleration_for_nongravitating_sph_particle(r, i, gb);
						particles[i].h *= 0.5*(1+cbrt(50./((double)particles[i].nn)));
						if (particles[i].h <= 0) particles[i].h = r->boxsize_max/2.;
						if (particles[i].h > r->boxsize_max) particles[i].h = r->boxsize_max/2.;	
						particles[i].rho += particles[i].rhoi;
						if (r->eos == REB_EOS_ISOTHERMAL) {
							double dx = gb.shiftx - particles[0].x;
							double dy = gb.shifty - particles[0].y;
							double dz = gb.shiftz - particles[0].z;
							double _r = sqrt(dx*dx + dy*dy + dz*dz);
							particles[i].cs = r->eos_isothermal.cs0 * pow(_r, r->eos_isothermal.q);
						} else {
							particles[i].cs = sqrt(r->hydro.gamma * particles[i].p / particles[i].rhoi);
						}
						newdt = MIN(newdt, tau*particles[i].h/particles[i].cs);
					}				
				}
				// Gravity
				struct reb_ghostbox gb = reb_boundary_get_ghostbox(r, gbx,gby,gbz);
#pragma omp parallel for
				for (int i=0; i<_N_real; i++){
#ifndef OPENMP
                if (reb_sigint) return;
#endif // OPENMP
				for (int j=0; j<_N_active; j++){
					// if (_gravity_ignore_terms==1 && ((j==1 && i==0) || (i==1 && j==0) )) continue;
					// if (_gravity_ignore_terms==2 && ((j==0 || i==0) )) continue;
					if (i==j) continue;
					const double dx = (gb.shiftx+particles[i].x) - particles[j].x;
					const double dy = (gb.shifty+particles[i].y) - particles[j].y;
					const double dz = (gb.shiftz+particles[i].z) - particles[j].z;
					const double _r = sqrt(dx*dx + dy*dy + dz*dz + softening2);
					const double prefact = -G/(_r*_r*_r)*particles[j].m;
					
					particles[i].ax    += prefact*dx;
					particles[i].ay    += prefact*dy;
					particles[i].az    += prefact*dz;
				}
				}
                if (_testparticle_type){
				for (int i=0; i<_N_active; i++){
#ifndef OPENMP
                if (reb_sigint) return;
#endif // OPENMP
				for (int j=_N_active; j<_N_real; j++){
					// if (_gravity_ignore_terms==1 && ((j==1 && i==0) )) continue;
					// if (_gravity_ignore_terms==2 && ((j==0 || i==0) )) continue;
					const double dx = (gb.shiftx+particles[i].x) - particles[j].x;
					const double dy = (gb.shifty+particles[i].y) - particles[j].y;
					const double dz = (gb.shiftz+particles[i].z) - particles[j].z;
					double softening = r->softening+particles[j].h;
					const double _r = sqrt(dx*dx + dy*dy + dz*dz + softening*softening);
					const double prefact = -G/(_r*_r*_r)*particles[j].m;
					
					particles[i].ax    += prefact*dx;
					particles[i].ay    += prefact*dy;
					particles[i].az    += prefact*dz;
				}
				}
                }
				
			}
			}
			}
			//Todo: make the dt more adjustable.
			r->dt = newdt; // If the number of sph particles are not many and when they are collaping, the dt may becomes -inf and N_tot is 0 (I don't know why it becomes zero.)
		}
		break;

        case REB_GRAVITY_MERCURIUS:
		break;

		default:
			reb_exit("Pressure term in momentum equation is not calculated.");
	}
}

// Helper routines for REB_GRAVITY_TREE


/**
  * @brief The function calls itself recursively using cell breaking criterion to check whether it can use center of mass (and mass quadrupole tensor) to calculate forces.
  * Calculate the acceleration for a particle from a given cell and all its daughter cells.
  *
  * @param r REBOUND simulation to consider
  * @param pt Index of the particle the force is calculated for.
  * @param node Pointer to the cell the force is calculated from.
  * @param gb Ghostbox plus position of the particle (precalculated). 
  */
static void reb_calculate_acceleration_for_sph_particle_from_cell(const struct reb_simulation* const r, const int pt, const struct reb_treecell *node, const struct reb_ghostbox gb);
  
static void reb_calculate_acceleration_for_sph_particle(const struct reb_simulation* const r, const int pt, const struct reb_ghostbox gb) {
	for(int i=0;i<r->root_n;i++){
		struct reb_treecell* node = r->tree_root[i];
			if (node!=NULL){
				reb_calculate_acceleration_for_sph_particle_from_cell(r, pt, node, gb);
			} else {
				printf("Node is NULL\n");
			}
	}
}

static double kernel(double nu, double h){
	if (nu >= 0. && nu < 1.) {
		return (1. - 1.5*nu*nu + 0.75*nu*nu*nu)/M_PI/h/h/h;
	} else if (nu >=1. && nu < 2.) {
		return (2. - nu)*(2.-nu)*(2.-nu)/4./M_PI/h/h/h;
	} else {
		return 0.;
	}
}

static double kernel_center(double h){
	return 1./M_PI/h/h/h;
}

static double kernel_derivative(double _r, double h){
	double h4 = h*h*h*h;
	if (_r <= h){
		return 3./M_PI/h4/h * (0.75*_r/h - 1.);
		// for(int i=0;i<3;i++) delW[i] = prefact * dr[i];
	} else if (_r <= 2.*h) {
		// double prefact = 3./4./M_PI/h/h/h/h/_r;
		return -0.75/M_PI/h4/_r * (2. - _r/h)*(2. - _r/h);
	} else {
		return 0;
	}
}

static void reb_calculate_acceleration_for_sph_particle_from_cell(const struct reb_simulation* r, const int pt, const struct reb_treecell *node, const struct reb_ghostbox gb) {
	const double G = r->G;
	struct reb_particle* const particles = r->particles;
	// double softening2;
	const double dx = gb.shiftx - node->mx;
	const double dy = gb.shifty - node->my;
	const double dz = gb.shiftz - node->mz;
	const double r2 = dx*dx + dy*dy + dz*dz;
	// softening2 = particles[pt].h * particles[pt].h;
	if ( node->pt < 0 ) { // Not a leaf
		const double dxij = particles[pt].x - node->x;
		const double dyij = particles[pt].y - node->y;
		const double dzij = particles[pt].z - node->z;
		const double thesdist = 2.*particles[pt].h + node->w/2.;	
		// if ( (node->w*node->w > r->opening_angle2*r2) || ( 4.*particles[pt].h*particles[pt].h > r2) ){
		if ( (node->w*node->w > r->opening_angle2*r2) || ( MAX(MAX(dxij*dxij, dyij*dyij), dzij*dzij) < thesdist*thesdist ) ){
			for (int o=0; o<8; o++) {
				if (node->oct[o] != NULL) {
					reb_calculate_acceleration_for_sph_particle_from_cell(r, pt, node->oct[o], gb);
				}
			}
		} else {
			double _r = sqrt(r2);
			double prefact = -G/(_r*_r*_r)*node->m;
#ifdef QUADRUPOLE
			double qprefact = G/(_r*_r*_r*_r*_r);
			particles[pt].ax += qprefact*(dx*node->mxx + dy*node->mxy + dz*node->mxz); 
			particles[pt].ay += qprefact*(dx*node->mxy + dy*node->myy + dz*node->myz); 
			particles[pt].az += qprefact*(dx*node->mxz + dy*node->myz + dz*node->mzz); 
			double mrr 	= dx*dx*node->mxx 	+ dy*dy*node->myy 	+ dz*dz*node->mzz
					+ 2.*dx*dy*node->mxy 	+ 2.*dx*dz*node->mxz 	+ 2.*dy*dz*node->myz; 
			qprefact *= -5.0/(2.0*_r*_r)*mrr;
			particles[pt].ax += (qprefact + prefact) * dx; 
			particles[pt].ay += (qprefact + prefact) * dy; 
			particles[pt].az += (qprefact + prefact) * dz; 
#else
			particles[pt].ax += prefact*dx; 
			particles[pt].ay += prefact*dy; 
			particles[pt].az += prefact*dz; 
#endif
		}
	} else { // It's a leaf node
		if (node->pt == pt) return;
		if ( r2 > 4.*particles[pt].h*particles[pt].h ) { // The node is not within the particle's kernel
			double _r = sqrt(r2);
			double prefact = -G/(_r*_r*_r)*node->m;
			particles[pt].ax += prefact*dx; 
			particles[pt].ay += prefact*dy; 
			particles[pt].az += prefact*dz;
		} else {
			// Gravity
			double softening = 0.5*(particles[pt].h + particles[node->pt].h);
			double _r = sqrt(r2 + softening*softening);
			double prefact = -G/(_r*_r*_r)*node->m;
			particles[pt].ax += prefact*dx; 
			particles[pt].ay += prefact*dy; 
			particles[pt].az += prefact*dz;
			// Pressure
			if (particles[node->pt].type == REB_PTYPE_SPH) {
				_r = sqrt(r2);
				const double dvx = particles[pt].vx - particles[node->pt].vx;
				const double dvy = particles[pt].vy - particles[node->pt].vy;
				const double dvz = particles[pt].vz - particles[node->pt].vz;
				const double dv = sqrt(dvx*dvx + dvy*dvy + dvz*dvz);	
				const double angle = acos((dx*dvx + dy*dvy + dz*dvz)/_r/dv);
				double p_e_prefact = node->m * (kernel_derivative(_r, particles[pt].h) + kernel_derivative(_r, particles[node->pt].h));
				double pprefact = - sqrt(particles[pt].p*particles[node->pt].p)/particles[pt].rhoi/particles[node->pt].rhoi * p_e_prefact;
				double eprefact = particles[pt].p/particles[pt].rhoi/particles[pt].rhoi * p_e_prefact/2.;
				particles[pt].ax += pprefact*dx;
				particles[pt].ay += pprefact*dy;
				particles[pt].az += pprefact*dz;
				if (angle <= M_PI/2.) {
					particles[pt].e += -eprefact*dv;
				} else {
					if (r->hydro.av == REB_HYDRO_ARTIFICIAL_VISCOSITY_ON) {
						double hij = (particles[pt].h + particles[node->pt].h)/2.0;
						double muij = (dvx*dx + dvy*dy + dvz*dz)/hij/(r2/hij/hij + 0.01);
						double visij = (-0.5*r->hydro.alpha*muij*(particles[pt].cs + particles[node->pt].cs) + r->hydro.beta*muij*muij)*2.0/(particles[pt].rhoi + particles[node->pt].rhoi);
						particles[pt].ax += -0.5*visij*p_e_prefact*dx;
						particles[pt].ay += -0.5*visij*p_e_prefact*dy;
						particles[pt].az += -0.5*visij*p_e_prefact*dz;
					}
					particles[pt].e += eprefact*dv;
				}
				
				particles[pt].rho += node->m * (kernel(_r/particles[node->pt].h, particles[node->pt].h) + kernel(_r/particles[node->pt].h, particles[pt].h))/2.;
				particles[pt].nn += 1;
			}
		}
	}
}

static void reb_calculate_acceleration_for_nbody_particle_from_cell(const struct reb_simulation* const r, const int pt, const struct reb_treecell *node, const struct reb_ghostbox gb);

static void reb_calculate_acceleration_for_nbody_particle(const struct reb_simulation* const r, const int pt, const struct reb_ghostbox gb) {
	for(int i=0;i<r->root_n;i++){
		struct reb_treecell* node = r->tree_root[i];
		if (node!=NULL){
			reb_calculate_acceleration_for_nbody_particle_from_cell(r, pt, node, gb);
		}
	}
}

static void reb_calculate_acceleration_for_nbody_particle_from_cell(const struct reb_simulation* r, const int pt, const struct reb_treecell *node, const struct reb_ghostbox gb) {
	const double G = r->G;
	const double softening2 = r->softening*r->softening;
	struct reb_particle* const particles = r->particles;
	const double dx = gb.shiftx - node->mx;
	const double dy = gb.shifty - node->my;
	const double dz = gb.shiftz - node->mz;
	const double r2 = dx*dx + dy*dy + dz*dz;
	if ( node->pt < 0 ) { // Not a leaf
		if ( node->w*node->w > r->opening_angle2*r2 ){
			for (int o=0; o<8; o++) {
				if (node->oct[o] != NULL) {
					reb_calculate_acceleration_for_nbody_particle_from_cell(r, pt, node->oct[o], gb);
				}
			}
		} else {
			double _r = sqrt(r2 + softening2);
			double prefact = -G/(_r*_r*_r)*node->m;
#ifdef QUADRUPOLE
			double qprefact = G/(_r*_r*_r*_r*_r);
			particles[pt].ax += qprefact*(dx*node->mxx + dy*node->mxy + dz*node->mxz); 
			particles[pt].ay += qprefact*(dx*node->mxy + dy*node->myy + dz*node->myz); 
			particles[pt].az += qprefact*(dx*node->mxz + dy*node->myz + dz*node->mzz); 
			double mrr 	= dx*dx*node->mxx 	+ dy*dy*node->myy 	+ dz*dz*node->mzz
					+ 2.*dx*dy*node->mxy 	+ 2.*dx*dz*node->mxz 	+ 2.*dy*dz*node->myz; 
			qprefact *= -5.0/(2.0*_r*_r)*mrr;
			particles[pt].ax += (qprefact + prefact) * dx; 
			particles[pt].ay += (qprefact + prefact) * dy; 
			particles[pt].az += (qprefact + prefact) * dz; 
#else
			particles[pt].ax += prefact*dx; 
			particles[pt].ay += prefact*dy; 
			particles[pt].az += prefact*dz; 
#endif
		}
	} else { // It's a leaf node
		if (node->pt == pt) return;
		double _r = sqrt(r2 + softening2);
		double prefact = -G/(_r*_r*_r)*node->m;
		particles[pt].ax += prefact*dx; 
		particles[pt].ay += prefact*dy; 
		particles[pt].az += prefact*dz; 
	}
}

static void reb_calculate_acceleration_for_nongravitating_sph_particle_from_cell(const struct reb_simulation* r, const int pt, const struct reb_treecell *node, const struct reb_ghostbox gb) {
	// const double G = r->G;
	struct reb_particle* const particles = r->particles;
	// double softening2;
	const double dx = gb.shiftx - node->x;
	const double dy = gb.shifty - node->y;
	const double dz = gb.shiftz - node->z;
	const double r2 = dx*dx + dy*dy + dz*dz;
	// softening2 = particles[pt].h * particles[pt].h;
	if ( node->pt < 0 ) { // Not a leaf
		const double thesdist = 2.*particles[pt].h + node->w/2.;	
		if ( MAX(MAX(dx*dx, dy*dy), dz*dz) < thesdist*thesdist ){
			for (int o=0; o<8; o++) {
				if (node->oct[o] != NULL) {
					reb_calculate_acceleration_for_nongravitating_sph_particle_from_cell(r, pt, node->oct[o], gb);
				}
			}
		}
	} else { // It's a leaf node
		if (node->pt == pt) return;
		if ( r2 <= 4.*particles[pt].h*particles[pt].h ) { 
			// // Gravity
			double softening = 0.5*(particles[pt].h + particles[node->pt].h);
			double _r = sqrt(r2 + softening*softening);
			// double prefact = -G/(_r*_r*_r)*node->m;
			// particles[pt].ax += prefact*dx; 
			// particles[pt].ay += prefact*dy; 
			// particles[pt].az += prefact*dz;
			// Pressure
			if (particles[node->pt].type == REB_PTYPE_SPH) {
				_r = sqrt(r2);
				const double dvx = particles[pt].vx - particles[node->pt].vx;
				const double dvy = particles[pt].vy - particles[node->pt].vy;
				const double dvz = particles[pt].vz - particles[node->pt].vz;
				const double dv = sqrt(dvx*dvx + dvy*dvy + dvz*dvz);	
				const double angle = acos((dx*dvx + dy*dvy + dz*dvz)/_r/dv);
				double p_e_prefact = particles[node->pt].m * (kernel_derivative(_r, particles[pt].h) + kernel_derivative(_r, particles[node->pt].h));
				double pprefact = - sqrt(particles[pt].p*particles[node->pt].p)/particles[pt].rhoi/particles[node->pt].rhoi * p_e_prefact;
				double eprefact = particles[pt].p/particles[pt].rhoi/particles[pt].rhoi * p_e_prefact/2.;
				particles[pt].ax += pprefact*dx; 
				particles[pt].ay += pprefact*dy;
				particles[pt].az += pprefact*dz;
				if (angle < M_PI/2.) {
					particles[pt].e += -eprefact*dv;
				} else {
					if (r->hydro.av == REB_HYDRO_ARTIFICIAL_VISCOSITY_ON) {
						double hij = (particles[pt].h + particles[node->pt].h)/2.0;
						double muij = (dvx*dx + dvy*dy + dvz*dz)/hij/(r2/hij/hij + 0.01);
						double visij = (-0.5*1.0*muij*(particles[pt].cs + particles[node->pt].cs) +2.0*muij*muij)*2.0/(particles[pt].rhoi + particles[node->pt].rhoi);
						particles[pt].ax += -0.5*visij*p_e_prefact*dx;
						particles[pt].ay += -0.5*visij*p_e_prefact*dy;
						particles[pt].az += -0.5*visij*p_e_prefact*dz;
					}
					particles[pt].e += eprefact*dv;
				}
				
				particles[pt].rho += particles[node->pt].m * (kernel(_r/particles[node->pt].h, particles[node->pt].h) + kernel(_r/particles[node->pt].h, particles[pt].h))/2.;
				particles[pt].nn += 1;
			}
		}
	}
}

static void reb_calculate_acceleration_for_nongravitating_sph_particle(const struct reb_simulation* const r, const int pt, const struct reb_ghostbox gb) {
	for(int i=0;i<r->root_n;i++){
		struct reb_treecell* node = r->tree_root[i];
			if (node!=NULL){
				reb_calculate_acceleration_for_nongravitating_sph_particle_from_cell(r, pt, node, gb);
			}
	}
}
