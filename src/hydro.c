/**
 * @file 	hydro.c
 * @brief 	Smooth particle hydrodynamics
 * @author 	Shangfei Liu <shangfei.liu@gmail.com>
 *
 * @details 	This is the crudest implementation of an N-body code
 * which sums up every pair of particles. It is only useful very small 
 * particle numbers (N<~100) as it scales as O(N^2). Note that the MPI
 * implementation is not well tested and only works for very specific
 * problems. This should be resolved in the future. 
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
static void reb_calculate_gravitational_acceleration_for_sph_particle(const struct reb_simulation* const r, const int pt, const struct reb_ghostbox gb);

static void reb_calculate_pressure_for_sph_particle(const struct reb_simulation* const r, const int pt);

static void reb_calculate_pressure_acceleration_for_sph_particle(const struct reb_simulation* const r, const int pt, const struct reb_ghostbox gb);


void reb_calculate_hydrodynamics(struct reb_simulation* r){
	struct reb_particle* const particles = r->particles;
	const int N = r->N;
	const int N_active = r->N_active;
	const double G = r->G;
	const double softening2 = r->softening*r->softening;
	const unsigned int _gravity_ignore_terms = r->gravity_ignore_terms;
	const int _N_active = ((N_active==-1)?N:N_active) - r->N_var;
	const int _N_real   = N  - r->N_var;
	const int _testparticle_type   = r->testparticle_type;
	switch (r->gravity){
		case REB_GRAVITY_NONE: // Do nothing.
		break;

		case REB_GRAVITY_BASIC:
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
				// particles[i].rho = 0;
				particles[i].nn = 0;
				particles[i].newrho = 0;
				// particles[i].p 	= 0; 
			}
			// Summing over all Ghost Boxes
			for (int gbx=-r->nghostx; gbx<=r->nghostx; gbx++){
			for (int gby=-r->nghosty; gby<=r->nghosty; gby++){
			for (int gbz=-r->nghostz; gbz<=r->nghostz; gbz++){
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
					reb_calculate_gravitational_acceleration_for_sph_particle(r, i, gb);
					// if (particles[i].rho != 0) {
					// 	double dh = - particles[i].h/particles[i].rho*(particles[i].newrho-particles[i].rho)/3.;
					// 	if (particles[i].nn >= 70) {
					// 		particles[i].h /= 2.;
					// 	} else if (particles[i].nn < 30) {
					// 		particles[i].h *= 2.;
					// 	} else {
					// 		particles[i].h += dh;
					// 	}
					// }
					particles[i].h *= 0.5*(1+cbrt(50./((double)particles[i].nn)));
					if (particles[i].h <= 0) particles[i].h = 2.e8;
					if (particles[i].h > 3.5e9) particles[i].h = 3.5e9;			
					particles[i].rho = particles[i].newrho;
					reb_calculate_pressure_for_sph_particle(r, i);								
				}

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
					reb_calculate_pressure_acceleration_for_sph_particle(r, i, gb);
				}

			}
			}
			}
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
static void reb_calculate_gravitational_acceleration_for_sph_particle_from_cell(const struct reb_simulation* const r, const int pt, const struct reb_treecell *node, const struct reb_ghostbox gb);
  
static void reb_calculate_gravitational_acceleration_for_sph_particle(const struct reb_simulation* const r, const int pt, const struct reb_ghostbox gb) {
	for(int i=0;i<r->root_n;i++){
		struct reb_treecell* node = r->tree_root[i];
			if (node!=NULL){
				reb_calculate_gravitational_acceleration_for_sph_particle_from_cell(r, pt, node, gb);
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

static void reb_calculate_gravitational_acceleration_for_sph_particle_from_cell(const struct reb_simulation* r, const int pt, const struct reb_treecell *node, const struct reb_ghostbox gb) {
	const double G = r->G;
	struct reb_particle* const particles = r->particles;
	double softening2;
	const double dx = gb.shiftx - node->mx;
	const double dy = gb.shifty - node->my;
	const double dz = gb.shiftz - node->mz;
	const double r2 = dx*dx + dy*dy + dz*dz;
	if ( particles[pt].h*2. <= MIN(MIN(particles[pt].c->w/2. - fabs(particles[pt].x-particles[pt].c->x), particles[pt].c->w/2. - fabs(particles[pt].y-particles[pt].c->y)), \
	  particles[pt].c->w/2. - fabs(particles[pt].z-particles[pt].c->z)) ) { // Cell size larger than smoothing sphere, no neighboring sph particles
		if ( node->pt < 0 ) { // Not a leaf
			if ( node->w*node->w > r->opening_angle2*r2 ){
				for (int o=0; o<8; o++) {
					if (node->oct[o] != NULL) {
						reb_calculate_gravitational_acceleration_for_sph_particle_from_cell(r, pt, node->oct[o], gb);
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
			if (node->pt == pt) {
				particles[pt].newrho += particles[pt].m * kernel_center(particles[pt].h);
				// reb_calculate_pressure_for_sph_particle(r, pt);
				return;
			}
			double _r = sqrt(r2);
			double prefact = -G/(_r*_r*_r)*node->m;
			particles[pt].ax += prefact*dx; 
			particles[pt].ay += prefact*dy; 
			particles[pt].az += prefact*dz; 
		}
	} else { // There are neighboring sph particles
		softening2 = particles[pt].h * particles[pt].h;
		if ( node->pt < 0 ) { // Not a leaf
			if ( (node->w*node->w > r->opening_angle2*r2) || (particles[pt].h*2. > MAX(MAX(fabs(dx), fabs(dy)), fabs(dz))) ){
				for (int o=0; o<8; o++) {
					if (node->oct[o] != NULL) {
						reb_calculate_gravitational_acceleration_for_sph_particle_from_cell(r, pt, node->oct[o], gb);
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
			if (node->pt == pt) { // The particle itself
				particles[pt].newrho += particles[pt].m * kernel_center(particles[pt].h);
				return;
			}
			if ( particles[pt].h*2. <= MAX(MAX(fabs(dx), fabs(dy)), fabs(dz)) ) { // The node is not within the particle's kernel
				double _r = sqrt(r2);
				double prefact = -G/(_r*_r*_r)*node->m;
				particles[pt].ax += prefact*dx; 
				particles[pt].ay += prefact*dy; 
				particles[pt].az += prefact*dz;
				// particles[pt].rho += node->m * (kernel(_r/particles[node->pt].h, particles[node->pt].h) + kernel(_r/particles[node->pt].h, particles[pt].h))/2.;
			} else {
				double _r = sqrt(r2 + softening2);
				double prefact = -G/(_r*_r*_r)*node->m;
				particles[pt].ax += prefact*dx; 
				particles[pt].ay += prefact*dy; 
				particles[pt].az += prefact*dz;
				particles[pt].newrho += node->m * (kernel(sqrt(r2)/particles[node->pt].h, particles[node->pt].h) + kernel(sqrt(r2)/particles[node->pt].h, particles[pt].h))/2.;
				particles[pt].nn += 1;
			}
		}
	}
}

static void reb_calculate_pressure_acceleration_for_sph_particle_from_cell(const struct reb_simulation* const r, const int pt, const struct reb_treecell *node, const struct reb_ghostbox gb);

static void reb_calculate_pressure_acceleration_for_sph_particle(const struct reb_simulation* const r, const int pt, const struct reb_ghostbox gb) {
	for(int i=0;i<r->root_n;i++){
		struct reb_treecell* node = r->tree_root[i];
			if (node!=NULL){
				reb_calculate_pressure_acceleration_for_sph_particle_from_cell(r, pt, node, gb);
			}
	}
}

static void reb_calculate_pressure_acceleration_for_sph_particle_from_cell(const struct reb_simulation* r, const int pt, const struct reb_treecell *node, const struct reb_ghostbox gb) {
	struct reb_particle* const particles = r->particles;
	const double dx = gb.shiftx - node->mx;
	const double dy = gb.shifty - node->my;
	const double dz = gb.shiftz - node->mz;
	const double r2 = dx*dx + dy*dy + dz*dz;
	if ( node->pt < 0 ) { // Not a leaf
		if ( particles[pt].h*2. > MAX(MAX(fabs(dx), fabs(dy)), fabs(dz)) ) {
			for (int o=0; o<8; o++) {
				if (node->oct[o] != NULL) {
					reb_calculate_pressure_acceleration_for_sph_particle_from_cell(r, pt, node->oct[o], gb);
				}
			}
		}
	} else { // It's a leaf node
		if (node->pt == pt) { // The particle itself
			// particles[pt].h *= 0.5*(1+cbrt(30./((double)particles[pt].nn)));
			return;
		}
		if ( particles[pt].h*2. > MAX(MAX(fabs(dx), fabs(dy)), fabs(dz)) ) { // The node is within the particle's kernel
			double _r = sqrt(r2);
			double pprefact = -node->m*sqrt(particles[pt].p*particles[node->pt].p)/particles[pt].rho/particles[node->pt].rho * (kernel_derivative(_r, particles[pt].h) + kernel_derivative(_r, particles[node->pt].h));
			particles[pt].ax += pprefact*dx; 
			particles[pt].ay += pprefact*dy;
			particles[pt].az += pprefact*dz;
		}
	}	
}

static void reb_calculate_pressure_for_sph_particle(const struct reb_simulation* const r, const int pt){
	struct reb_particle* const particles = r->particles;
	double K = 2.6e12; // dyne g^-2 cm^4
	particles[pt].p = K*particles[pt].rho*particles[pt].rho;	
}
