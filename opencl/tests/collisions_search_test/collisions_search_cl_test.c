#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define M_PI 3.141592653589793238462643383279502884197169339375

#include "cl_gpu_defns.h"
#include "../../src/cl_host_tools.h"
#include "collisions_search_cl_test.h"
#include "cpu_collisions_search.h"

/*

  THINGS TO DO:
   ++++POTENTIAL OPTIMIZATION AND CLARIFICATION OF CODE: WRITE ALL NEEDED BUFFERS TO GPU BEFORE KERNEL CALLS JUST TO MAKE SURE EVERYTHING IS ON DEVICE BEFORE ITERATIONS

*/

void collisions_search_cl_test(int num_bodies, int num_threads_tree_kernel, int num_threads_tree_gravity_kernel, int num_threads_tree_sort_kernel, int num_threads_force_gravity_kernel, int num_threads_collisions_search_kernel, int num_threads_collisions_resolve_kernel)
{
  cl_device_id device;
  cl_context context;
  cl_program program;
  cl_kernel tree_kernel;
  cl_kernel tree_gravity_kernel;
  cl_kernel tree_sort_kernel;
  cl_kernel force_gravity_kernel;
  cl_kernel collisions_search_kernel;
  cl_kernel collisions_resolve_kernel;
  cl_command_queue queue;
  size_t local_size_tree_kernel;
  size_t global_size_tree_kernel;
  size_t local_size_tree_gravity_kernel;
  size_t global_size_tree_gravity_kernel;
  size_t local_size_tree_sort_kernel;
  size_t global_size_tree_sort_kernel;
  size_t local_size_force_gravity_kernel;
  size_t wave_fronts_in_force_gravity_kernel;
  size_t global_size_force_gravity_kernel;
  size_t local_size_collisions_search_kernel;
  size_t wave_fronts_in_collisions_search_kernel;
  size_t global_size_collisions_search_kernel;
  size_t local_size_collisions_resolve_kernel;
  size_t global_size_collisions_resolve_kernel;
  cl_int error;
  cl_int work_groups;
  cl_mem error_buffer;
  cl_mem x_buffer;
  cl_mem y_buffer;
  cl_mem z_buffer;
  cl_mem cellcenter_x_buffer;
  cl_mem cellcenter_y_buffer;
  cl_mem cellcenter_z_buffer;
  cl_mem vx_buffer;
  cl_mem vy_buffer;
  cl_mem vz_buffer;
  cl_mem ax_buffer;
  cl_mem ay_buffer;
  cl_mem az_buffer;
  cl_mem t_buffer;
  cl_mem rad_buffer;
  cl_mem mass_buffer;
  cl_mem start_buffer;
  cl_mem sort_buffer;
  cl_mem count_buffer;
  cl_mem children_buffer;
  cl_mem collisions_buffer;
  cl_mem bottom_node_buffer;
  cl_mem softening2_buffer;
  cl_mem inv_opening_angle2_buffer;
  cl_mem minimum_collision_velocity_buffer;
  cl_mem maxdepth_buffer;
  cl_mem boxsize_buffer;
  cl_mem rootx_buffer;
  cl_mem rooty_buffer;
  cl_mem rootz_buffer;
  cl_mem collisions_max2_r_buffer;
  cl_mem num_nodes_buffer;
  cl_mem num_bodies_buffer;
  cl_mem OMEGA_buffer;
  cl_int *children_host;
  cl_int *collisions_host;
  cl_float *x_host;
  cl_float *y_host;
  cl_float *z_host;
  cl_float *cellcenter_x_host;
  cl_float *cellcenter_y_host;
  cl_float *cellcenter_z_host;
  cl_float *vx_host;
  cl_float *vy_host;
  cl_float *vz_host;
  cl_float *ax_host;
  cl_float *ay_host;
  cl_float *az_host;
  cl_float *mass_host;
  cl_int *count_host;
  cl_int *sort_host;
  cl_int *start_host;
  cl_float *rad_host;
  cl_int nghostx;
  cl_int nghosty;
  cl_int nghostz;
  cl_float t_host;
  cl_float softening2_host;
  cl_float inv_opening_angle2_host;
  cl_float minimum_collision_velocity_host;
  cl_float collisions_max_r_host;
  cl_float collisions_max2_r_host;
  cl_int bottom_node_host;
  cl_int maxdepth_host;
  cl_int num_nodes_host;
  cl_int num_bodies_host;
  cl_float boxsize_host;
  cl_float rootx_host;
  cl_float rooty_host;			      
  cl_float rootz_host;

  device = cl_host_tools_create_device();
  
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
 
  //srand(time(NULL));
  srand(123213444);

  t_host = 0.f;
  cl_float surfacedensity = 400; //kg/m^2
  cl_float particle_density = 400; // kg/m^3
  cl_float particle_radius_min = 1; //m
  cl_float particle_radius_max = 4;
  cl_float particle_radius_slope = -3;
  boxsize_host = 100;

  // num_bodies_host = num_bodies;
  //boxsize_host = 1;
  rootx_host = 0.f;
  rooty_host = 0.f;
  rootz_host = 0.f;

  cl_float OMEGA_host = 0.00013143527f;
  //cl_float dt = 1e-3*2.*M_PI/OMEGA_host;
  minimum_collision_velocity_host = particle_radius_min*OMEGA_host*0.001; 
  softening2_host = 0.1*0.1;
  inv_opening_angle2_host = 4;

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
    printf("rad_host_temp[%d] = %.10f\n",p, rad_host_temp[p]);
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
    rad_host[i] = rad_host_temp[i];
    mass_host[i] = mass_host_temp[i];
  }

  printf(" work items per workgroup in tree kernel = %u\n", (unsigned int)local_size_tree_kernel);
  printf(" work items per workgroup in tree gravity kernel = %u\n", (unsigned int)local_size_tree_gravity_kernel);
  printf(" work items per workgroup in tree sort kernel = %u\n", (unsigned int)local_size_tree_sort_kernel);
  printf(" work items per workgroup in force gravity kernel = %u\n", (unsigned int)local_size_force_gravity_kernel);
  printf(" work items per workgroup in collisions search kernel = %u\n", (unsigned int)local_size_collisions_search_kernel);
  printf(" work items per workgroup in collisions resolve kernel = %u\n", (unsigned int)local_size_collisions_resolve_kernel);
  printf(" work groups = %u\n", (unsigned int)work_groups);
  printf(" total work items in tree kernel = %u\n", (unsigned int)global_size_tree_kernel);
  printf(" total work items in tree gravity kernel = %u\n", (unsigned int)global_size_tree_gravity_kernel);
  printf(" total work items in tree sort kernel = %u\n", (unsigned int)global_size_tree_sort_kernel);
  printf(" total work items in force gravity kernel = %u\n", (unsigned int)global_size_force_gravity_kernel);
  printf(" total wave fronts per work group in force gravity kernel= %u\n", (unsigned int)wave_fronts_in_force_gravity_kernel);
  printf(" total work items in collision search kernel = %u\n", (unsigned int)global_size_collisions_search_kernel);
  printf(" total wave fronts per work group in collision search kernel= %u\n", (unsigned int)wave_fronts_in_collisions_search_kernel);
  printf(" total work items in collision resolve kernel = %u\n", (unsigned int)global_size_collisions_resolve_kernel);

  printf(" num_nodes_host = %d\n", num_nodes_host);
  printf(" num_bodies_host = %d\n", num_bodies_host);

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateContext ERROR");
    exit(EXIT_FAILURE);
  }

  const char *options = "-cl-single-precision-constant -cl-opt-disable";
  const char *file_names [] = {"../../src/cl_tree.cl", 
			       "../../src/cl_gravity_tree.cl", 
			       "../../src/cl_boundaries_shear.cl",
			       "../../src/cl_collisions_tree.cl" 
			       };
  program = cl_host_tools_create_program(context, device, file_names, options, 4);

  /* const char *options = ""; */
  /* const char *file_names [] = {"../../src/cl_tree.cl"}; */
  /* program = cl_host_tools_create_program(context, device, file_names, options, 1); */

  /* const char *options = ""; */
  /* const char *file_names [] = {"../../src/test.cl"}; */
  /* program = cl_host_tools_create_program(context, device, file_names, options, 1); */

  // Create buffers

  error_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &error);
  if (error != CL_SUCCESS){
    fprintf(stderr, "clCreateBuffer Error (error_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  x_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (num_nodes_host+1) * sizeof(cl_float), x_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (x_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  y_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (num_nodes_host+1) * sizeof(cl_float), y_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (y_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  z_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (num_nodes_host+1) * sizeof(cl_float), z_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (z_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  cellcenter_x_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, (num_nodes_host+1) * sizeof(cl_float), NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (cellcenter_x_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  cellcenter_y_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, (num_nodes_host+1) * sizeof(cl_float), NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (cellcenter_y_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  cellcenter_z_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, (num_nodes_host+1) * sizeof(cl_float), NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (cellcenter_z_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  vx_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (num_bodies_host) * sizeof(cl_float), vx_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (vx_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  vy_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (num_bodies_host) * sizeof(cl_float), vy_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (vy_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  vz_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (num_bodies_host) * sizeof(cl_float), vz_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (vz_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  ax_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, num_bodies_host * sizeof(cl_float), NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (x_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  ay_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, num_bodies_host * sizeof(cl_float), NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (y_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  az_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, num_bodies_host * sizeof(cl_float), NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (z_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }  

   mass_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (num_nodes_host+1) * sizeof(cl_float), mass_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (mass_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  rad_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (num_bodies_host) * sizeof(cl_float), rad_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (rad_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  start_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, (num_nodes_host+1) * sizeof(cl_int), NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (start_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  sort_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, (num_nodes_host+1) * sizeof(cl_int), NULL, &error);
  if (error != CL_SUCCESS){
    fprintf(stderr, "clCreateBuffer ERROR (sort_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  count_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, (num_nodes_host+1) * sizeof(cl_int), NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr, "clCreateBuffer ERROR (count_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  children_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 8 * (num_nodes_host+1) * sizeof(cl_int), NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (children_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  collisions_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * num_bodies_host * (nghostx + 1) * (nghosty + 1) * (nghostz + 1), NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr, "clCreateBuffer ERROR (collisions_buffer): %s\n", cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  bottom_node_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (bottom_node_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  softening2_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &softening2_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr, "clCreateBuffer Error (softening2_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  inv_opening_angle2_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &inv_opening_angle2_host, &error);
  if (error != CL_SUCCESS){
    fprintf(stderr, "clCreateBuffer Error (inv_opening_angle2_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  minimum_collision_velocity_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &minimum_collision_velocity_host, &error);
  if (error != CL_SUCCESS){
    fprintf(stderr, "clCreateBuffer Error (minimum_collision_velocity): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  maxdepth_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (maxdepth_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  num_nodes_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_int), &num_nodes_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (num_nodes_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  num_bodies_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_int), &num_bodies_host,  &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (num_bodies_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  boxsize_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &boxsize_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (boxsize_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  rootx_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &rootx_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (rootx_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  rooty_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &rooty_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (rooty_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  rootz_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &rootz_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (rootz_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
 OMEGA_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &OMEGA_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (OMEGA_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
 t_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &t_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (OMEGA_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  collisions_max2_r_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &collisions_max2_r_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (collisions_max2_r_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  //Create a command queue
  queue = clCreateCommandQueue(context, device, 0, &error);
  if (error != CL_SUCCESS){
    fprintf(stderr,"clCreateCommandQueue ERROR: %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  };

  //Create kernels
  tree_kernel = clCreateKernel(program, "cl_tree_add_particles_to_tree", &error);
  if (error != CL_SUCCESS){
    fprintf(stderr,"clCreateKernel ERROR (tree_kernel): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  tree_gravity_kernel = clCreateKernel(program, "cl_tree_update_tree_gravity_data", &error);
  if (error != CL_SUCCESS){
    fprintf(stderr, "clCreateKernel ERROR (tree_gravity_kernel): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  tree_sort_kernel = clCreateKernel(program, "cl_tree_sort_particles", &error);
  if (error != CL_SUCCESS){
    fprintf(stderr, "clCreateKernel ERROR (tree_sort_kernel): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  force_gravity_kernel = clCreateKernel(program, "cl_gravity_calculate_acceleration_for_particle", &error);
  if (error != CL_SUCCESS){
    fprintf(stderr, "clCreateKernel ERROR (tree_force_gravity_kernel): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  collisions_search_kernel = clCreateKernel(program, "cl_collisions_search", &error);
  if (error != CL_SUCCESS){
    fprintf(stderr, "clCreateKernel ERROR (collisions_search_kernel): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  collisions_resolve_kernel = clCreateKernel(program, "cl_collisions_resolve", &error);
  if (error != CL_SUCCESS){
    fprintf(stderr, "clCreateKernel ERROR (collisions_resolve_kernel): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  //Set Kernel Arguments for tree_kernel  
  error = clSetKernelArg(tree_kernel, 0, sizeof(cl_mem), &x_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/x_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 1, sizeof(cl_mem), &y_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/y_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 2, sizeof(cl_mem), &z_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/z_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 3, sizeof(cl_mem), &mass_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/mass_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 4, sizeof(cl_mem), &start_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/start_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 5, sizeof(cl_mem), &children_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/children_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 6, sizeof(cl_mem), &maxdepth_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/maxdepth_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 7, sizeof(cl_mem), &bottom_node_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/bottom_node_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 8, sizeof(cl_mem), &boxsize_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/boxsize_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 9, sizeof(cl_mem), &rootx_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/rootx_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 10, sizeof(cl_mem), &rooty_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/rooty_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 11, sizeof(cl_mem), &rootz_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/rootz_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 12, sizeof(cl_mem), &num_nodes_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/num_nodes_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 13, sizeof(cl_mem), &num_bodies_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/num_bodies_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  //Set Kernel Arguments for tree_gravity_kernel
  error = clSetKernelArg(tree_gravity_kernel, 0, sizeof(cl_mem), &x_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/x_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 1, sizeof(cl_mem), &y_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/y_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 2, sizeof(cl_mem), &z_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/z_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 3, sizeof(cl_mem), &cellcenter_x_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/cellcenter_x_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 4, sizeof(cl_mem), &cellcenter_y_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/cellcenter_y_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 5, sizeof(cl_mem), &cellcenter_z_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/cellcenter_z_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 6, sizeof(cl_mem), &mass_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/mass_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 7, sizeof(cl_mem), &children_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/children_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 8, sizeof(cl_mem), &count_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/count_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 9, sizeof(cl_mem), &bottom_node_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/bottom_node_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }  
  error = clSetKernelArg(tree_gravity_kernel, 10, sizeof(cl_mem), &num_nodes_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/num_nodes_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 11, sizeof(cl_mem), &num_bodies_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/num_bodies_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 12, sizeof(cl_int)*8*num_threads_tree_gravity_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/children_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  } 

  //Set Kernel Arguments for tree_sort_kernel
  error = clSetKernelArg(tree_sort_kernel, 0, sizeof(cl_mem), &children_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_sort_kernel/children_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_sort_kernel, 1, sizeof(cl_mem), &count_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_sort_kernel/count_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_sort_kernel, 2, sizeof(cl_mem), &start_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_sort_kernel/start_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_sort_kernel, 3, sizeof(cl_mem), &sort_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_sort_kernel/sort_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_sort_kernel, 4, sizeof(cl_mem), &bottom_node_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_sort_kernel/bottom_node_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }  
  error = clSetKernelArg(tree_sort_kernel, 5, sizeof(cl_mem), &num_nodes_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_sort_kernel/num_nodes_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_sort_kernel, 6, sizeof(cl_mem), &num_bodies_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_sort_kernel/num_bodies_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  //Set Kernel Arguments for force_gravity_kernel
  error = clSetKernelArg(force_gravity_kernel, 0, sizeof(cl_mem), &x_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/x_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 1, sizeof(cl_mem), &y_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/y_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 2, sizeof(cl_mem), &z_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/z_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 3, sizeof(cl_mem), &ax_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/ax_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 4, sizeof(cl_mem), &ay_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/ay_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 5, sizeof(cl_mem), &az_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/az_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 6, sizeof(cl_mem), &mass_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/mass_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 7, sizeof(cl_mem), &sort_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/sort_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
    error = clSetKernelArg(force_gravity_kernel, 8, sizeof(cl_mem), &children_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/children_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 9, sizeof(cl_mem), &maxdepth_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/maxdepth_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 10, sizeof(cl_mem), &boxsize_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/boxsize_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
   error = clSetKernelArg(force_gravity_kernel, 11, sizeof(cl_mem), &num_nodes_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/num_nodes_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 12, sizeof(cl_mem), &num_bodies_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/num_bodies_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 13, sizeof(cl_mem), &inv_opening_angle2_buffer);
  if (error != CL_SUCCESS){
    fprintf(stderr, "clSetKernelArg ERROR (force_gravity_kernel/inv_opening_angle2_buffer: %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 14, sizeof(cl_mem), &softening2_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/softening2_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 15, sizeof(cl_int)*wave_fronts_in_force_gravity_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/children_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 16, sizeof(cl_int)*wave_fronts_in_force_gravity_kernel*MAX_DEPTH, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/pos_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 17, sizeof(cl_int)*wave_fronts_in_force_gravity_kernel*MAX_DEPTH, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/node_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 18, sizeof(cl_float)*wave_fronts_in_force_gravity_kernel*MAX_DEPTH, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/dr_cutoff_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 19, sizeof(cl_float)*wave_fronts_in_force_gravity_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/nodex_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 20, sizeof(cl_float)*wave_fronts_in_force_gravity_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/nodey_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 21, sizeof(cl_float)*wave_fronts_in_force_gravity_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/nodez_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 22, sizeof(cl_float)*wave_fronts_in_force_gravity_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/nodem_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 23, sizeof(cl_int)*local_size_force_gravity_kernel, NULL);			 
  if (error != CL_SUCCESS) {
    fprintf(stderr, "clSetKernelArg Error (force_gravity_kernel/wavefront_vote_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

//Set Kernel Arguments for collisions_search_kernel
  error = clSetKernelArg(collisions_search_kernel, 0, sizeof(cl_mem), &x_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/x_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 1, sizeof(cl_mem), &y_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/y_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 2, sizeof(cl_mem), &z_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/z_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 3, sizeof(cl_mem), &cellcenter_x_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/cellcenter_x_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 4, sizeof(cl_mem), &cellcenter_y_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/cellcenter_y_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 5, sizeof(cl_mem), &cellcenter_z_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/cellcenter_z_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 6, sizeof(cl_mem), &vx_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/vx_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 7, sizeof(cl_mem), &vy_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/vy_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 8, sizeof(cl_mem), &vz_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/vz_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 9, sizeof(cl_mem), &mass_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/mass_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 10, sizeof(cl_mem), &rad_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/mass_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 11, sizeof(cl_mem), &sort_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/sort_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
    error = clSetKernelArg(collisions_search_kernel, 12, sizeof(cl_mem), &collisions_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/collisions_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
    error = clSetKernelArg(collisions_search_kernel, 13, sizeof(cl_mem), &children_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/children_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 14, sizeof(cl_mem), &maxdepth_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/maxdepth_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 15, sizeof(cl_mem), &boxsize_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/boxsize_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 16, sizeof(cl_mem), &num_nodes_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/num_nodes_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 17, sizeof(cl_mem), &num_bodies_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/num_bodies_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 18, sizeof(cl_int)*wave_fronts_in_collisions_search_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/children_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 19, sizeof(cl_int)*wave_fronts_in_collisions_search_kernel*MAX_DEPTH, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/pos_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 20, sizeof(cl_int)*wave_fronts_in_collisions_search_kernel*MAX_DEPTH, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/node_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 21, sizeof(cl_float)*wave_fronts_in_collisions_search_kernel*MAX_DEPTH, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/dr_cutoff_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 22, sizeof(cl_float)*wave_fronts_in_collisions_search_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/nodex_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 23, sizeof(cl_float)*wave_fronts_in_collisions_search_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/nodey_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 24, sizeof(cl_float)*wave_fronts_in_collisions_search_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/nodez_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel,25, sizeof(cl_float)*wave_fronts_in_collisions_search_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/nodevx_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel,26, sizeof(cl_float)*wave_fronts_in_collisions_search_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/nodevy_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 27, sizeof(cl_float)*wave_fronts_in_collisions_search_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/nodevz_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 28, sizeof(cl_float)*wave_fronts_in_collisions_search_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/noderad_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 29, sizeof(cl_int)*local_size_collisions_search_kernel, NULL);			
  if (error != CL_SUCCESS) {
    fprintf(stderr, "clSetKernelArg Error (collisions_search_kernel/wavefront_vote_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 30, sizeof(cl_mem), &collisions_max2_r_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/collisions_max2_r_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 31, sizeof(cl_mem), &OMEGA_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/OMEGA_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 32, sizeof(cl_mem), &t_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/t_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  //Set Kernel Arguments for collisions_resolve kernel
  error = clSetKernelArg(collisions_resolve_kernel, 0, sizeof(cl_mem), &x_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_resolve_kernel/x_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_resolve_kernel, 1, sizeof(cl_mem), &y_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_resolve_kernel/y_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_resolve_kernel, 2, sizeof(cl_mem), &z_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_resolve_kernel/z_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_resolve_kernel, 3, sizeof(cl_mem), &vx_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_resolve_kernel/vx_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_resolve_kernel, 4, sizeof(cl_mem), &vy_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_resolve_kernel/vy_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }  
  error = clSetKernelArg(collisions_resolve_kernel, 5, sizeof(cl_mem), &vz_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_resolve_kernel/vz_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_resolve_kernel, 6, sizeof(cl_mem), &mass_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_resolve_kernel/mass_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_resolve_kernel, 7, sizeof(cl_mem), &rad_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_resolve_kernel/rad_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_resolve_kernel, 8, sizeof(cl_mem), &sort_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_resolve_kernel/sort_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_resolve_kernel, 9, sizeof(cl_mem), &collisions_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_resolve_kernel/collisions_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_resolve_kernel, 10, sizeof(cl_mem), &num_bodies_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_resolve_kernel/num_bodies_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_resolve_kernel, 11, sizeof(cl_mem), &minimum_collision_velocity_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_resolve_kernel/minimum_collision_velocity_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_resolve_kernel, 12, sizeof(cl_mem), &OMEGA_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_resolve_kernel/OMEGA_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_resolve_kernel, 13, sizeof(cl_mem), &boxsize_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_resolve_kernel/boxsize_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_resolve_kernel, 14, sizeof(cl_mem), &t_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_resolve_kernel/t_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_resolve_kernel, 15, sizeof(cl_mem), &error_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_resolve_kernel/error_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  //Set kernel arguments for boundaries kernel


  //Enqueue kernels
  error = clEnqueueNDRangeKernel(queue, tree_kernel, 1, 0, &global_size_tree_kernel, &local_size_tree_kernel, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueNDRangeKernel ERROR (tree_kernel): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  
  error = clEnqueueNDRangeKernel(queue, tree_gravity_kernel, 1, 0, &global_size_tree_gravity_kernel, &local_size_tree_gravity_kernel, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueNDRangeKernel ERROR (tree_gravity_kernel): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueNDRangeKernel(queue, tree_sort_kernel, 1, 0, &global_size_tree_sort_kernel, &local_size_tree_sort_kernel, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueNDRangeKernel ERROR (tree_sort_kernel): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
 
   error = clEnqueueNDRangeKernel(queue, force_gravity_kernel, 1, 0, &global_size_force_gravity_kernel, &local_size_force_gravity_kernel, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr, "clEqueueNDRangeKernel ERROR (force_gravity_kernel): %s\n", cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueNDRangeKernel(queue, collisions_search_kernel, 1, 0, &global_size_collisions_search_kernel, &local_size_collisions_search_kernel, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr, "clEqueueNDRangeKernel ERROR (collisions_search_kernel): %s\n", cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueNDRangeKernel(queue, collisions_resolve_kernel, 1, 0, &global_size_collisions_resolve_kernel, &local_size_collisions_resolve_kernel, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr, "clEqueueNDRangeKernel ERROR (collisions_resolve_kernel): %s\n", cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  //Block until queue is done
  error = clFinish(queue);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clFinish ERROR: %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  // Read the results
  error = clEnqueueReadBuffer(queue, children_buffer, CL_TRUE, 0, sizeof(cl_int) * 8 * (num_nodes_host + 1), children_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (children_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, collisions_buffer, CL_TRUE, 0, sizeof(cl_int) * num_bodies_host * (nghostx + 1) * (nghosty + 1) * (nghostz + 1), collisions_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (collisions_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, bottom_node_buffer, CL_TRUE, 0, sizeof(cl_int), &bottom_node_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (bottom_node_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, maxdepth_buffer, CL_TRUE, 0, sizeof(cl_int), &maxdepth_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (maxdepth_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, mass_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_nodes_host + 1), mass_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (mass_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, start_buffer, CL_TRUE, 0, sizeof(cl_int) * (num_nodes_host + 1), start_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (start_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, sort_buffer, CL_TRUE, 0, sizeof(cl_int) * (num_nodes_host + 1), sort_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (sort_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, x_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_nodes_host + 1), x_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (x_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, y_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_nodes_host + 1), y_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (y_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, z_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_nodes_host + 1), z_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (z_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, cellcenter_x_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_nodes_host + 1), cellcenter_x_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (cellcenter_x_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, cellcenter_y_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_nodes_host + 1), cellcenter_y_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (cellcenter_y_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, cellcenter_z_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_nodes_host + 1), cellcenter_z_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (cellcenter_z_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  
  error = clEnqueueReadBuffer(queue, count_buffer, CL_TRUE, 0, sizeof(cl_int) * (num_nodes_host + 1), count_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (count_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, ax_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_bodies_host), ax_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr, "clEqueueReadBuffer ERROR (ax_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, ay_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_bodies_host), ay_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr, "clEqueueReadBuffer ERROR (ay_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, az_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_bodies_host), az_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr, "clEqueueReadBuffer ERROR (az_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, vx_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_bodies_host), vx_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr, "clEqueueReadBuffer ERROR (vx_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, vy_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_bodies_host), vy_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr, "clEqueueReadBuffer ERROR (vy_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, vz_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_bodies_host), vz_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr, "clEqueueReadBuffer ERROR (vz_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }


  float error_host;

  error = clEnqueueReadBuffer(queue, error_buffer,CL_TRUE, 0, sizeof(cl_float), &error_host, 0, NULL, NULL);
  if(error != CL_SUCCESS){
    fprintf(stderr, "clEnqueueReadBuffer ERROR (error_buffer: %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  printf("\n++++++ERROR+++++++\n");
  printf("error_host = %.6f", error_host);

  printf("\n++++++TREE++++++\n");

  for (int node = bottom_node_host; node < num_nodes_host + 1; node++){
    printf("+++++NODE %d+++++:",node);
    for (int child = 0; child < 8; child++)
      printf(" %d ", children_host[node*8 + child]);
    printf("\n");
  }
  printf("\n++++++X+Y+Z+AX+AY+AZ+MASS++++++\n");
  
  cl_float com_x = 0.f;
  cl_float com_y = 0.f;
  cl_float com_z = 0.f;

  for (int i = 0; i < num_nodes_host + 1; i++){
    if (i < num_bodies_host){
      printf(" %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", i, x_host[i], y_host[i], z_host[i], vx_host[i], vy_host[i], vz_host[i], vx_host_temp[i], vy_host_temp[i], vz_host_temp[i], ax_host[i], ay_host[i], az_host[i],  mass_host[i], rad_host[i]);
      com_x += mass_host[i]*x_host[i];
      com_y += mass_host[i]*y_host[i];
      com_z += mass_host[i]*z_host[i];
    }
    else
      printf(" %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", i, x_host[i], y_host[i], z_host[i], 0., 0., 0.,  mass_host[i]);      
  }

  printf("\n Total Mass = %.6f", total_mass);
  printf("\n Total COM = %.6f %.6f %.6f\n", com_x, com_y, com_z);

  printf("\n++++++CELL_CENTERS++++++\n");
  for (int i = num_nodes_host; i >= bottom_node_host; i--){
    printf(" %d %.6f %.6f %.6f \n", i, cellcenter_x_host[i], cellcenter_y_host[i] , cellcenter_z_host[i]);
  }

  /* printf("\n++++++CELL_CENTERS++++++\n"); */
  /* printf("\n++++++STORED+++++THEORETICAL++++++\n"); */
  
  /* cl_float * cenx_host; */
  /* cl_float * ceny_host; */
  /* cl_float * cenz_host; */
  /* cl_float * radius_host; */
  /* cenx_host = (cl_float *) malloc( sizeof(cl_float) * ((num_nodes_host) + 1) ); */
  /* ceny_host = (cl_float *) malloc( sizeof(cl_float) * ((num_nodes_host) + 1) ); */
  /* cenz_host = (cl_float *) malloc( sizeof(cl_float) * ((num_nodes_host) + 1) ); */
  /* radius_host = (cl_float *) malloc( sizeof(cl_float) * ((num_nodes_host) + 1) ); */

  /* cenx_host[num_nodes_host] = rootx_host; */
  /* ceny_host[num_nodes_host] = rooty_host; */
  /* cenz_host[num_nodes_host] = rootz_host; */
  /* radius_host[num_nodes_host] = boxsize_host/2.; */
  /* cl_float tempx = 0; */
  /* cl_float tempy = 0; */
  /* cl_float tempz = 0; */
  /* cl_int child_temp; */
  /* int z = num_nodes_host; */
  /* printf(" %d %.6f %.6f %.6f  %.6f %.6f %.6f \n", num_nodes_host, cellcenter_x_host[z], cellcenter_y_host[z], cellcenter_z_host[z], cenx_host[z], ceny_host[z], cenz_host[z]);  */
  /* for (int i = num_nodes_host; i >= bottom_node_host; i--){ */
  /*   for (int j = 0; j < 8; j++){ */
  /*     child_temp = children_host[i*8 + j]; */
  /*     if (child_temp >= num_bodies_host){ */
  /* 	tempx = (j & 1) * radius_host[i]; */
  /* 	tempy = ((j >> 1) & 1) * radius_host[i]; */
  /* 	tempz = ((j >> 2) & 1) * radius_host[i]; */
  /* 	radius_host[child_temp] = radius_host[i]*.5f; */
  /* 	cenx_host[child_temp] = cenx_host[i] - radius_host[child_temp] + tempx; */
  /* 	ceny_host[child_temp] = ceny_host[i] - radius_host[child_temp] + tempy; */
  /* 	cenz_host[child_temp] = cenz_host[i] - radius_host[child_temp] + tempz; */
  /* 	printf(" %d %.6f %.6f %.6f %.6f %.6f %.6f \n", child_temp, cellcenter_x_host[child_temp], cellcenter_y_host[child_temp] , cellcenter_z_host[child_temp], cenx_host[child_temp],ceny_host[child_temp], cenz_host[child_temp]); */
  /*     } */
  /*   } */
  /* } */

  /* free(cenx_host); */
  /* free(ceny_host); */
  /* free(cenz_host); */
  /* free(radius_host); */
  
  printf("\n++++++COUNT_HOST++++++\n");
  for (int i = 0; i < num_nodes_host + 1; i++){
    printf(" %d %d \n", i, count_host[i]);
  }

  printf("\n++++++SORT_HOST++++++\n");
  for (int i = 0; i < num_nodes_host + 1; i++){
    printf(" %d %d \n", i, sort_host[i]);
  }

  printf("\n++++++START_HOST++++++\n"); 
  for (int i = 0; i < num_nodes_host + 1; i++){
    printf(" %d %d \n", i, start_host[i]);
  }

  printf("\n++++++COLLISIONS_HOST+++++++\n");
  for (int i = 0; i < num_bodies_host * (nghostx + 1) * (nghosty + 1) * (nghostz + 1); i++){
    printf(" %d %d \n", i, collisions_host[i]);
  }

  printf("\n bottom_node_host = %d\n", (int)bottom_node_host);
  printf("maxdepth_host = %d\n", (int)maxdepth_host);
  printf("collisions_max2_r = %.10f\n", (float)collisions_max2_r_host);

  ///////////////////////////////// CPU //////////////////////////////////////////////////////
  #ifdef CPU_TEST
  cl_int* children_local;
  cl_int* pos_local;
  cl_int* node_local;
  
  cl_float* dr_cutoff_local;
  cl_float* nodex_local;
  cl_float* nodey_local;
  cl_float* nodez_local;
  cl_float* nodem_local;
  cl_int* wavefront_vote_local;

  children_local = (cl_int *) malloc( sizeof(cl_int) * wave_fronts_in_force_gravity_kernel);
  pos_local =(cl_int *) malloc( sizeof(cl_int) * wave_fronts_in_force_gravity_kernel*MAX_DEPTH);
  node_local =(cl_int *) malloc( sizeof(cl_int) * wave_fronts_in_force_gravity_kernel*MAX_DEPTH);
  dr_cutoff_local = (cl_float *) malloc( sizeof(cl_float) * wave_fronts_in_force_gravity_kernel*MAX_DEPTH);
  nodex_local = (cl_float *) malloc( sizeof(cl_float) * wave_fronts_in_force_gravity_kernel);
  nodey_local = (cl_float *) malloc( sizeof(cl_float) * wave_fronts_in_force_gravity_kernel);
  nodez_local = (cl_float *) malloc( sizeof(cl_float) * wave_fronts_in_force_gravity_kernel);
  nodem_local = (cl_float *) malloc( sizeof(cl_float) * wave_fronts_in_force_gravity_kernel);
  wavefront_vote_local = (cl_int *) malloc( sizeof(cl_int) * wave_fronts_in_force_gravity_kernel);

  cpu_gravity_calculate_acceleration_for_particle(x_host, y_host, z_host, ax_host, ay_host, az_host, mass_host, sort_host, children_host, &maxdepth_host, &bottom_node_host, &boxsize_host, &num_nodes_host, &num_bodies_host, &inv_opening_angle2_host, &softening2_host, children_local, pos_local, node_local, dr_cutoff_local, nodex_local, nodey_local, nodez_local, nodem_local, wavefront_vote_local, (int) wave_fronts_in_force_gravity_kernel);

  printf("\n++++++AFTER CPU++++++\n");
  for (int i = 0; i < num_nodes_host + 1; i++){
    if (i < num_bodies_host)
      printf(" %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", i, x_host[i], y_host[i], z_host[i], ax_host[i], ay_host[i], az_host[i],  mass_host[i]);
    else
      printf(" %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", i, x_host[i], y_host[i], z_host[i], 0., 0., 0.,  mass_host[i]);
  }

  free(children_local);
  free(pos_local);
  free(node_local);
  free(dr_cutoff_local);
  free(nodex_local);
  free(nodey_local);
  free(nodez_local);
  free(nodem_local);
  free(wavefront_vote_local);
  #endif
  //////////////////////////////
  ///CPU COLLISIONS TEST
  /////////////////////////////
  //#define CPU_COLLISIONS_TEST
#ifdef CPU_COLLISIONS_TEST

  //////////////////////////////////////

  cl_int* children_local;
  cl_int* pos_local;
  cl_int* node_local;
  
  cl_float* dr_cutoff_local;
  cl_float* nodex_local;
  cl_float* nodey_local;
  cl_float* nodez_local;
  cl_float* nodevx_local;
  cl_float* nodevy_local;
  cl_float* nodevz_local;
  cl_float* nodem_local;
  cl_float* noderad_local;
  cl_int* wavefront_vote_local;
  cl_int * collisions_host_cpu2;
  collisions_host_cpu2 = (cl_int *) malloc( sizeof(cl_int) * num_bodies_host * (nghostx + 1) * (nghosty + 1) * (nghostz + 1) );

  children_local = (cl_int *) malloc( sizeof(cl_int) * wave_fronts_in_collisions_search_kernel);
  pos_local =(cl_int *) malloc( sizeof(cl_int) * wave_fronts_in_collisions_search_kernel*MAX_DEPTH);
  node_local =(cl_int *) malloc( sizeof(cl_int) * wave_fronts_in_collisions_search_kernel*MAX_DEPTH);
  dr_cutoff_local = (cl_float *) malloc( sizeof(cl_float) * wave_fronts_in_collisions_search_kernel*MAX_DEPTH);
  nodex_local = (cl_float *) malloc( sizeof(cl_float) * wave_fronts_in_collisions_search_kernel);
  nodey_local = (cl_float *) malloc( sizeof(cl_float) * wave_fronts_in_collisions_search_kernel);
  nodez_local = (cl_float *) malloc( sizeof(cl_float) * wave_fronts_in_collisions_search_kernel);
  nodevx_local = (cl_float *) malloc( sizeof(cl_float) * wave_fronts_in_collisions_search_kernel);
  nodevy_local = (cl_float *) malloc( sizeof(cl_float) * wave_fronts_in_collisions_search_kernel);
  nodevz_local = (cl_float *) malloc( sizeof(cl_float) * wave_fronts_in_collisions_search_kernel);
  nodem_local = (cl_float *) malloc( sizeof(cl_float) * wave_fronts_in_collisions_search_kernel);
  noderad_local = (cl_float *) malloc( sizeof(cl_float) * wave_fronts_in_collisions_search_kernel);
  wavefront_vote_local = (cl_int *) malloc( sizeof(cl_int) * wave_fronts_in_collisions_search_kernel*WAVEFRONT_SIZE);

  cpu_collisions_search_2(x_host, y_host, z_host, cellcenter_x_host, cellcenter_y_host, cellcenter_z_host, vx_host, vy_host, vz_host, mass_host, rad_host, sort_host, collisions_host_cpu2, children_host, &maxdepth_host, &boxsize_host, &num_nodes_host, &num_bodies_host, children_local, pos_local, node_local, dr_cutoff_local, nodex_local, nodey_local, nodez_local, nodevx_local, nodevy_local, nodevz_local, nodem_local, noderad_local, wavefront_vote_local, &collisions_max2_r_host, &OMEGA_host, &t_host, (cl_int) wave_fronts_in_collisions_search_kernel);

  ////////////////////////////////////  
  cl_int * collisions_host_cpu;

  collisions_host_cpu = (cl_int *) malloc( sizeof(cl_int) * num_bodies_host * (nghostx + 1) * (nghosty + 1) * (nghostz + 1) );

  cpu_collisions_search(x_host,
			y_host,
			z_host,
			rad_host,
			vx_host,
			vy_host,
			vz_host,
			collisions_host_cpu,
			&num_bodies_host,
			&nghostx, 
			&nghosty,
			&nghostz,
			&boxsize_host,
			&OMEGA_host,
			&t_host);

  int gbx,gby,gbz,b;
  int gbx_offset = num_bodies_host;
  int gby_offset = gbx_offset*3;
  int gbz_offset = gby_offset*3;
  printf("\n++++++AFTER CPU++++++\n");
  printf("\n++++++COLLISIONS_HOST_CPU, COLLISIONS_HOST+++++++\n");
  int col_error = 0;
  for (int i = 0; i < num_bodies_host * (nghostx + 1) * (nghosty + 1) * (nghostz + 1); i++){
    gbz = (cl_int)floor((float)i/(float)gbz_offset);
    gby = (cl_int)floor((float)(i - gbz*gbz_offset)/(float)(gby_offset)); 
    gbx = (cl_int)floor((float)(i - gbz*gbz_offset - gby*gby_offset)/(float)(gbx_offset));
    b = i - gbz*gbz_offset - gby*gby_offset - gbx*gbx_offset;
    gbx -= 1;
    gby -= 1;
    gbz -= 1;
    printf(" %d %d %d %d %d %d %d %d\n", i, b, gbx, gby, gbz, collisions_host_cpu[i], collisions_host_cpu2[i], collisions_host[i]);
    if (collisions_host_cpu[i] != collisions_host[i]){
      printf("\n +++ERROR DETECTED \n");
      float shiftvx = 0.f;
      float shiftvy = -1.5f*(float)gbx*(OMEGA_host)*(boxsize_host);
      float shiftvz = 0.f;
      float shift = (gbx == 0) ? 0.f : -fmod(shiftvy*(t_host) - ((gbx>0) - (gbx<0))*(boxsize_host)/2.f, boxsize_host) -  ((gbx>0) - (gbx<0))*(boxsize_host)/2.f;
      float shiftx = boxsize_host*(float)gbx;
      float shifty = boxsize_host*(float)gby-shift;
      float shiftz = boxsize_host*(float)gbz;
      int pid = collisions_host[i];
      float x1 = x_host[b] + shiftx;
      float y1 = y_host[b] + shifty;
      float z1 = z_host[b] + shiftz;
      float vx1 = vx_host[b] + shiftvx;
      float vy1 = vy_host[b] + shiftvy;
      float vz1 = vz_host[b] + shiftvz;
      float rad1 = rad_host[b];
      float x2 = x_host[pid];
      float y2 = y_host[pid];
      float z2 = z_host[pid];
      float vx2 = vx_host[pid];
      float vy2 = vy_host[pid];
      float vz2 = vz_host[pid];
      float rad2 = rad_host[pid];      
      float dx = (x1-x2);
      float dy = (y1-y2);
      float dz = (z1-z2);
      float dvx = (vx1-vx2);
      float dvy = (vy1-vy2);
      float dvz = (vz1-vz2);

      printf(" (gbx, gby, gbz, body1, body2) = (%d, %d, %d, %d, %d)\n", gbx, gby, gbz, b, pid);
      printf(" body1 (x,y,z) = ( %.6f , %.6f , %.6f )\n ", x1, y1, z1 );
      printf(" body2 (x,y,z) = ( %.6f , %.6f , %.6f )\n ", x2, y2, z2 );
      printf(" body1 (vx,vy,vz) = ( %.6f , %.6f , %.6f )\n ", vx1, vy1, vz1 );
      printf(" body2 (vx,vy,vz) = ( %.6f , %.6f , %.6f )\n ", vx2, vy2, vz2 );
      printf(" dr^2 = %.6f\n ", dx*dx + dy*dy + dz*dz);
      printf(" rp*rp = %.6f\n ", (rad1+rad2)*(rad1+rad2));
      printf(" dr dot dv = %.6f\n\n", dx*dvx + dy*dvy + dz*dvz);
      col_error += 1;
    }
  }  
  printf("error in collisions_search kernel = %d", col_error);
  free(collisions_host_cpu);
  free(collisions_host_cpu2);
  free(children_local);
  free(pos_local);
  free(node_local);
  free(dr_cutoff_local);
  free(nodevx_local);
  free(nodevy_local);
  free(nodevz_local);
  free(nodevx_local);
  free(nodevy_local);
  free(nodevz_local);
  free(nodem_local);
  free(noderad_local);
  free(wavefront_vote_local);


#endif

  /////////////////////////////


  free(x_host_temp);
  free(y_host_temp);
  free(z_host_temp);
  free(vx_host_temp);
  free(vy_host_temp);
  free(vz_host_temp);
  free(rad_host_temp);
  free(mass_host_temp);
  free(children_host);
  free(collisions_host);
  free(mass_host);
  free(count_host);
  free(start_host);
  free(sort_host);
  free(vx_host);
  free(vy_host);
  free(vz_host);
  free(ax_host);
  free(ay_host);
  free(az_host);
  free(x_host);
  free(y_host);
  free(z_host);
  free(cellcenter_x_host);
  free(cellcenter_y_host);
  free(cellcenter_z_host);

  clReleaseMemObject(x_buffer);
  clReleaseMemObject(y_buffer);
  clReleaseMemObject(z_buffer);
  clReleaseMemObject(cellcenter_x_buffer);
  clReleaseMemObject(cellcenter_y_buffer);
  clReleaseMemObject(cellcenter_z_buffer);
  clReleaseMemObject(vx_buffer);
  clReleaseMemObject(vy_buffer);
  clReleaseMemObject(vz_buffer);
  clReleaseMemObject(ax_buffer);
  clReleaseMemObject(ay_buffer);
  clReleaseMemObject(az_buffer);
  clReleaseMemObject(inv_opening_angle2_buffer);
  clReleaseMemObject(softening2_buffer);
  clReleaseMemObject(minimum_collision_velocity_buffer);
  clReleaseMemObject(mass_buffer);
  clReleaseMemObject(start_buffer);
  clReleaseMemObject(sort_buffer);
  clReleaseMemObject(count_buffer);
  clReleaseMemObject(collisions_buffer);
  clReleaseMemObject(children_buffer);
  clReleaseMemObject(bottom_node_buffer);
  clReleaseMemObject(maxdepth_buffer);
  clReleaseMemObject(boxsize_buffer);
  clReleaseMemObject(num_nodes_buffer);
  clReleaseMemObject(num_bodies_buffer);
  clReleaseMemObject(rootx_buffer);
  clReleaseMemObject(rooty_buffer);
  clReleaseMemObject(rootz_buffer);
  clReleaseMemObject(error_buffer);
  clReleaseKernel(tree_kernel);
  clReleaseKernel(tree_gravity_kernel);
  clReleaseKernel(tree_sort_kernel);
  clReleaseKernel(force_gravity_kernel);
  clReleaseKernel(collisions_search_kernel);
  clReleaseKernel(collisions_resolve_kernel);
  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);

}
