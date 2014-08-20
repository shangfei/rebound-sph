#ifndef CL_TESTS_H
#define CL_TESTS_H

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

void cl_tests_read_particle_positions_from_device();
void cl_tests_read_particle_velocities_from_device();
void cl_tests_read_particle_accelerations_from_device();
void cl_tests_read_particle_masses_from_device();
void cl_tests_read_tree_vars_from_device();
void cl_tests_print_particle_info(int, cl_char *, int);
void cl_tests_print_tree_info(int, cl_char *, int);
void cl_tests_cpu_force_gravity_direct_summation(cl_double *, cl_double *, cl_double *);
void cl_tests_force_gravity_test();
void cl_tests_boundaries_get_ghostbox(cl_int, cl_int, cl_int, cl_float *, cl_float *, cl_float *, cl_float *, cl_float *, cl_float *);
void cl_tests_collisions_search_test();
void cl_tests_cpu_collisions_search_direct(cl_int *);
#endif
