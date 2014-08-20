#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#ifndef __APPLE__
#define M_PI 3.141592653589793238462643383279502884197169339375
#endif

#include "cl_globals.h"
#include "cl_init.h"
#include "cl_tests.h"
#include "cl_gpu_defns.h"
#include "cl_host_tools.h"
#include "cl_shearing_sheet.h"

void cl_problem_init(int num_bodies, int num_threads_tree_kernel, int num_threads_tree_gravity_kernel, int num_threads_tree_sort_kernel, int num_threads_force_gravity_kernel, int num_threads_collisions_search_kernel, int num_threads_collisions_resolve_kernel, int num_threads_boundaries_kernel, int num_threads_integrator_kernel)
{

  cl_init_create_device();
  
  work_groups = cl_host_tools_get_num_compute_units(device);

  local_size_tree_kernel = num_threads_tree_kernel;
  global_size_tree_kernel = work_groups*local_size_tree_kernel;

  local_size_tree_gravity_kernel = num_threads_tree_gravity_kernel;
  global_size_tree_gravity_kernel = work_groups*local_size_tree_gravity_kernel;

  local_size_tree_sort_kernel = num_threads_tree_sort_kernel;
  global_size_tree_sort_kernel = work_groups*local_size_tree_sort_kernel;

  local_size_force_gravity_kernel = num_threads_force_gravity_kernel;
  global_size_force_gravity_kernel = work_groups*local_size_force_gravity_kernel;
 
  wave_fronts_in_force_gravity_kernel = local_size_force_gravity_kernel/WAVEFRONT_SIZE;

  local_size_collisions_search_kernel = num_threads_collisions_search_kernel;
  global_size_collisions_search_kernel = work_groups*local_size_collisions_search_kernel;
 
  wave_fronts_in_collisions_search_kernel = local_size_collisions_search_kernel/WAVEFRONT_SIZE;
  local_size_collisions_resolve_kernel = num_threads_collisions_resolve_kernel;
  global_size_collisions_resolve_kernel = work_groups*local_size_collisions_resolve_kernel;
 
  local_size_boundaries_kernel = num_threads_boundaries_kernel;
  global_size_boundaries_kernel = work_groups*local_size_boundaries_kernel;

  local_size_integrator_kernel = num_threads_integrator_kernel;
  global_size_integrator_kernel = work_groups*local_size_integrator_kernel;
  
  //srand(time(NULL));
  srand(123213444);

  t_host = 0.f;
  surfacedensity = 400; //kg/m^2
  particle_density = 400; // kg/m^3
  particle_radius_min = 1; //m
  particle_radius_max = 4; //m
  particle_radius_slope = -3;
  boxsize_host = 100;
  rootx_host = 0.f;
  rooty_host = 0.f;
  rootz_host = 0.f;
  OMEGA_host = 0.00013143527f;
  OMEGAZ_host = -1;
  dt_host = 1e-3*2.*M_PI/OMEGA_host;
  G_host = 6.67428e-11; // N / (1e-5 kg)^2 m^2
  
  cl_host_tools_integrator_cache_coefficients(&OMEGA_host,
					      &OMEGAZ_host,
					      &sindt_host,
					      &tandt_host,
					      &sindtz_host,
					      &tandtz_host,
					      &dt_host);

  minimum_collision_velocity_host = particle_radius_min*OMEGA_host*0.001; 
  softening2_host = 0.1*0.1;
  inv_opening_angle2_host = 100;

  cl_int p = 0;
  cl_float total_mass = surfacedensity*boxsize_host*boxsize_host;
  cl_float mass = 0;

  cl_int num_bodies_host_max = (int)ceil(total_mass/(particle_density*4./3.*M_PI* particle_radius_min * particle_radius_min * particle_radius_min));

  cl_float * x_host_temp;
  cl_float * y_host_temp;
  cl_float * z_host_temp;
  cl_float * vx_host_temp;
  cl_float * vy_host_temp;
  cl_float * vz_host_temp;
  cl_float * mass_host_temp;
  cl_float * rad_host_temp;

  x_host_temp = (cl_float *) malloc ( num_bodies_host_max * sizeof(cl_float));
  y_host_temp = (cl_float *) malloc ( num_bodies_host_max * sizeof(cl_float));
  z_host_temp = (cl_float *) malloc ( num_bodies_host_max * sizeof(cl_float));
  vx_host_temp = (cl_float *) malloc ( num_bodies_host_max * sizeof(cl_float));
  vy_host_temp = (cl_float *) malloc ( num_bodies_host_max * sizeof(cl_float));
  vz_host_temp = (cl_float *) malloc ( num_bodies_host_max * sizeof(cl_float));
  rad_host_temp = (cl_float *) malloc ( num_bodies_host_max * sizeof(cl_float));
  mass_host_temp = (cl_float *) malloc ( num_bodies_host_max * sizeof(cl_float));

  collisions_max_r_host = -1;
  collisions_max2_r_host = -1;

  while(mass<total_mass || p % WAVEFRONT_SIZE != 0){
    x_host_temp[p] = cl_host_tools_uniform(-boxsize_host/2.f,boxsize_host/2.f) + rootx_host;
    y_host_temp[p] = cl_host_tools_uniform(-boxsize_host/2.f,boxsize_host/2.f) + rooty_host;
    z_host_temp[p] = cl_host_tools_normal(1.) + rootz_host;				       
    vx_host_temp[p] = 0.f;
    vy_host_temp[p] = -1.5*x_host_temp[p]*OMEGA_host;
    vz_host_temp[p] = 0.f;
    rad_host_temp[p] = cl_host_tools_powerlaw(particle_radius_min,particle_radius_max,particle_radius_slope); // m
    /* printf("rad_host_temp[%d] = %.10f\n",p, rad_host_temp[p]); */
    mass_host_temp[p] = particle_density*4./3.*M_PI*rad_host_temp[p]*rad_host_temp[p]*rad_host_temp[p]; 	// kg
    mass += mass_host_temp[p];
    if (rad_host_temp[p] > collisions_max_r_host){
      collisions_max2_r_host = collisions_max_r_host;
      collisions_max_r_host = rad_host_temp[p];
    }
    else
      if (rad_host_temp[p] > collisions_max2_r_host)
	collisions_max2_r_host = rad_host_temp[p];
    p++;
  }
  p--; 
  total_mass = mass;
  num_bodies_host = p+1;
  
  //each leaf belongs to one parent node, so we at least need space for num_bodies of nodes
  //and we will need space for num_bodies of bodies, so:
  num_nodes_host = num_bodies_host * 2;
  if (num_nodes_host < 1024*work_groups)
    num_nodes_host = 1024*work_groups;
  while (num_nodes_host % WAVEFRONT_SIZE != 0) (num_nodes_host) ++;

  //we will be using num_nodes to retrieve the last array element, so we must decrement.
  (num_nodes_host)--;

  nghostx = 2;
  nghosty = 2;
  nghostz = 2;

  collisions_host = (cl_int *) malloc( sizeof(cl_int) * num_bodies_host * (nghostx + 1) * (nghosty + 1) * (nghostz + 1) );
  children_host = (cl_int *) malloc( sizeof(cl_int) * ((num_nodes_host) + 1) * 8);
  x_host = (cl_float *) malloc ( (num_nodes_host + 1) * sizeof(cl_float));
  y_host = (cl_float *) malloc ( (num_nodes_host + 1) * sizeof(cl_float));
  z_host = (cl_float *) malloc ( (num_nodes_host + 1) * sizeof(cl_float));
  cellcenter_x_host = (cl_float *) malloc ( (num_nodes_host + 1) * sizeof(cl_float));
  cellcenter_y_host = (cl_float *) malloc ( (num_nodes_host + 1) * sizeof(cl_float));
  cellcenter_z_host = (cl_float *) malloc ( (num_nodes_host + 1) * sizeof(cl_float));
  ax_host = (cl_float *) malloc ( num_bodies_host * sizeof(cl_float));
  ay_host = (cl_float *) malloc ( num_bodies_host * sizeof(cl_float));
  az_host = (cl_float *) malloc ( num_bodies_host * sizeof(cl_float));
  vx_host = (cl_float *) malloc ( num_bodies_host * sizeof(cl_float));
  vy_host = (cl_float *) malloc ( num_bodies_host * sizeof(cl_float));
  vz_host = (cl_float *) malloc ( num_bodies_host * sizeof(cl_float));
  rad_host = (cl_float *) malloc ( num_bodies_host * sizeof(cl_float));
  mass_host = (cl_float *) malloc ( (num_nodes_host + 1) * sizeof(cl_float));
  count_host = (cl_int *) malloc ( (num_nodes_host + 1) * sizeof(cl_int));
  sort_host = (cl_int *) malloc ( (num_nodes_host + 1) * sizeof(cl_int));
  start_host = (cl_int *) malloc ( (num_nodes_host + 1) * sizeof(cl_int));

  for (int i = 0; i < num_bodies_host; i++){
    x_host[i] = x_host_temp[i];
    y_host[i] = y_host_temp[i];
    z_host[i] = z_host_temp[i];
    vx_host[i] = vx_host_temp[i];
    vy_host[i] = vy_host_temp[i];
    vz_host[i] = vz_host_temp[i];
    ax_host[i] = -1.0f;
    ay_host[i] = -1.0f;
    az_host[i] = -1.0f;
    rad_host[i] = rad_host_temp[i];
    mass_host[i] = mass_host_temp[i];
  }

  free(x_host_temp);
  free(y_host_temp);
  free(z_host_temp);
  free(vx_host_temp);
  free(vy_host_temp);
  free(vz_host_temp);
  free(rad_host_temp);
  free(mass_host_temp);

  cl_init_print_thread_info();
  cl_init_create_context();
  cl_init_create_program();
  cl_init_create_buffers();
  cl_init_create_command_queue();
  cl_init_create_kernels();

  cl_init_set_kernel_arg_tree_kernel();
  cl_init_set_kernel_arg_tree_gravity_kernel();
  cl_init_set_kernel_arg_tree_sort_kernel();
  cl_init_set_kernel_arg_force_gravity_kernel();
  cl_init_set_kernel_arg_collisions_search_kernel();
  cl_init_set_kernel_arg_collisions_resolve_kernel();
  cl_init_set_kernel_arg_tree_collisions_kernel();
  cl_init_set_kernel_arg_tree_kernel_no_mass();
  cl_init_set_kernel_arg_boundaries_kernel();
  cl_init_set_kernel_arg_integrator_part1_kernel();
  cl_init_set_kernel_arg_integrator_part2_kernel();

  //#define TEST_KERNELS
#ifdef TEST_KERNELS
  cl_tests_force_gravity_test();
  cl_tests_collisions_search_test();
#endif

  clock_t begin, end;
  double time_spent;
  begin = clock();

  int steps = 1;
  for (int i = 0; i < steps; i++)  {
    cl_init_enqueue_integrator_part1_kernel();
    cl_init_enqueue_boundaries_kernel();
    cl_init_enqueue_tree_kernel();
    cl_init_enqueue_tree_gravity_kernel();
    cl_init_enqueue_tree_sort_kernel();
    cl_init_enqueue_force_gravity_kernel();
    cl_init_enqueue_integrator_part2_kernel();
    cl_init_enqueue_boundaries_kernel();
    cl_init_enqueue_tree_kernel_no_mass();
    cl_init_enqueue_tree_collisions_kernel();
    cl_init_enqueue_tree_sort_kernel();
    cl_init_enqueue_collisions_search_kernel();
    cl_init_enqueue_collisions_resolve_kernel();
  }

  //Block until queue is done
  error = clFinish(queue);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clFinish ERROR: %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf(" Execution time (s) = %f\n\n ", time_spent);
  
  cl_tests_print_particle_info(-1, (cl_char*) "\n PARTICLE INFO AFTER SIMULATION \n", 1);
  cl_init_free_globals();
 }
