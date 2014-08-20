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

#include "cl_globals.h"
#include "cl_init.h"
#include "cl_host_tools.h"
#include "cl_gpu_defns.h"

void cl_init_create_device(){
 device = cl_host_tools_create_device();
}

void cl_init_create_program(){
  const char *options = "-cl-single-precision-constant";
  const char *file_names [] = {
    "../src/cl_boundaries_shear.cl",
    "../src/cl_tree.cl", 
    "../src/cl_gravity_tree.cl", 
    "../src/cl_collisions_tree.cl", 
    "../src/cl_integrator_sei.cl"
  };
  program = cl_host_tools_create_program(context, device, file_names, options, 5);
}

void cl_init_create_context(){
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateContext ERROR");
    exit(EXIT_FAILURE);
  }
}

void cl_init_create_command_queue(){
  //Create a command queue
  queue = clCreateCommandQueue(context, device, 0, &error);
  if (error != CL_SUCCESS){
    fprintf(stderr,"clCreateCommandQueue ERROR: %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  };
}

void cl_init_create_kernels(){

  //Create kernels
  tree_kernel = clCreateKernel(program, "cl_tree_add_particles_to_tree", &error);
  if (error != CL_SUCCESS){
    fprintf(stderr,"clCreateKernel ERROR (tree_kernel): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  tree_kernel_no_mass = clCreateKernel(program, "cl_tree_add_particles_to_tree_no_mass", &error);
  if (error != CL_SUCCESS){
     fprintf(stderr, "clCreateKernel ERROR (tree_kernel_no_mass): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  tree_gravity_kernel = clCreateKernel(program, "cl_tree_update_tree_gravity_data", &error);
  if (error != CL_SUCCESS){
    fprintf(stderr, "clCreateKernel ERROR (tree_gravity_kernel): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  tree_collisions_kernel = clCreateKernel(program, "cl_tree_update_tree_collisions_data", &error);
  if (error != CL_SUCCESS){
    fprintf(stderr, "clCreateKernel ERROR (tree_collisions_kernel): %s\n",cl_host_tools_get_error_string(error));
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

  boundaries_kernel = clCreateKernel(program, "cl_boundaries_check", &error);
  if (error != CL_SUCCESS){
    fprintf(stderr, "clCreateKernel ERROR (cl_boundaries_check): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  integrator_part1_kernel = clCreateKernel(program, "cl_integrator_part1", &error);
  if (error != CL_SUCCESS){
    fprintf(stderr, "clCreateKernel ERROR (cl_integrator_part1): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  integrator_part2_kernel = clCreateKernel(program, "cl_integrator_part2", &error);
  if (error != CL_SUCCESS){
    fprintf(stderr, "clCreateKernel ERROR (cl_integrator_part2): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

}


void cl_init_print_thread_info(){

  printf(" work items per workgroup in tree kernel = %u\n", (unsigned int)local_size_tree_kernel);
  printf(" work items per workgroup in tree gravity kernel = %u\n", (unsigned int)local_size_tree_gravity_kernel);
  printf(" work items per workgroup in tree sort kernel = %u\n", (unsigned int)local_size_tree_sort_kernel);
  printf(" work items per workgroup in force gravity kernel = %u\n", (unsigned int)local_size_force_gravity_kernel);
  printf(" work items per workgroup in collisions search kernel = %u\n", (unsigned int)local_size_collisions_search_kernel);
  printf(" work items per workgroup in collisions resolve kernel = %u\n", (unsigned int)local_size_collisions_resolve_kernel);
  printf(" work items per workgroup in boundaries kernel = %u\n", (unsigned int)local_size_boundaries_kernel);
  printf(" work items per workgroup in integrator kernel(s) = %u\n", (unsigned int)local_size_boundaries_kernel);
  printf(" work groups = %u\n", (unsigned int)work_groups);
  printf(" total work items in tree kernel = %u\n", (unsigned int)global_size_tree_kernel);
  printf(" total work items in tree gravity kernel = %u\n", (unsigned int)global_size_tree_gravity_kernel);
  printf(" total work items in tree sort kernel = %u\n", (unsigned int)global_size_tree_sort_kernel);
  printf(" total work items in force gravity kernel = %u\n", (unsigned int)global_size_force_gravity_kernel);
  printf(" total wave fronts per work group in force gravity kernel= %u\n", (unsigned int)wave_fronts_in_force_gravity_kernel);
  printf(" total work items in collision search kernel = %u\n", (unsigned int)global_size_collisions_search_kernel);
  printf(" total wave fronts per work group in collision search kernel= %u\n", (unsigned int)wave_fronts_in_collisions_search_kernel);
  printf(" total work items in collision resolve kernel = %u\n", (unsigned int)global_size_collisions_resolve_kernel);
  printf(" total work items in boundaries kernel = %u\n", (unsigned int)global_size_boundaries_kernel);
  printf(" total work items in integrator kernel(s) = %u\n", (unsigned int)global_size_integrator_kernel);
  printf(" num_nodes_host = %d\n", num_nodes_host);
  printf(" num_bodies_host = %d\n", num_bodies_host);

}


void cl_init_create_buffers(){

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
  OMEGAZ_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &OMEGAZ_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (OMEGAZ_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  sindt_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &sindt_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (sindt_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  tandt_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &tandt_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (tandt_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  sindtz_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &sindtz_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (sindtz_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  tandtz_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &tandtz_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (tandtz_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  t_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &t_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (t_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  collisions_max2_r_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &collisions_max2_r_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (collisions_max2_r_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  dt_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &dt_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (dt_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  G_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &G_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (dt_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

}

void cl_init_set_kernel_arg_tree_kernel(){
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
}

void cl_init_set_kernel_arg_tree_gravity_kernel(){
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
  error = clSetKernelArg(tree_gravity_kernel, 3, sizeof(cl_mem), &mass_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/mass_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 4, sizeof(cl_mem), &children_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/children_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 5, sizeof(cl_mem), &count_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/count_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 6, sizeof(cl_mem), &bottom_node_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/bottom_node_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }  
  error = clSetKernelArg(tree_gravity_kernel, 7, sizeof(cl_mem), &num_nodes_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/num_nodes_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 8, sizeof(cl_mem), &num_bodies_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/num_bodies_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 9, sizeof(cl_int)*8*local_size_tree_gravity_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/children_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  } 
}

void cl_init_set_kernel_arg_tree_sort_kernel(){

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

}

void cl_init_set_kernel_arg_force_gravity_kernel(){

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
  error = clSetKernelArg(force_gravity_kernel, 10, sizeof(cl_mem), &t_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/t_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 11, sizeof(cl_mem), &OMEGA_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/OMEGA_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 12, sizeof(cl_mem), &boxsize_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/boxsize_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 13, sizeof(cl_mem), &num_nodes_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/num_nodes_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 14, sizeof(cl_mem), &num_bodies_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/num_bodies_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 15, sizeof(cl_mem), &inv_opening_angle2_buffer);
  if (error != CL_SUCCESS){
    fprintf(stderr, "clSetKernelArg ERROR (force_gravity_kernel/inv_opening_angle2_buffer: %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 16, sizeof(cl_mem), &softening2_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/softening2_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 17, sizeof(cl_mem), &G_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/softening2_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 18, sizeof(cl_int)*wave_fronts_in_force_gravity_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/children_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 19, sizeof(cl_int)*wave_fronts_in_force_gravity_kernel*MAX_DEPTH, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/pos_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 20, sizeof(cl_int)*wave_fronts_in_force_gravity_kernel*MAX_DEPTH, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/node_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 21, sizeof(cl_float)*wave_fronts_in_force_gravity_kernel*MAX_DEPTH, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/dr_cutoff_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 22, sizeof(cl_float)*wave_fronts_in_force_gravity_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/nodex_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 23, sizeof(cl_float)*wave_fronts_in_force_gravity_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/nodey_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 24, sizeof(cl_float)*wave_fronts_in_force_gravity_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/nodez_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 25, sizeof(cl_float)*wave_fronts_in_force_gravity_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (force_gravity_kernel/nodem_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(force_gravity_kernel, 26, sizeof(cl_int)*local_size_force_gravity_kernel, NULL);			 
  if (error != CL_SUCCESS) {
    fprintf(stderr, "clSetKernelArg Error (force_gravity_kernel/wavefront_vote_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

}


void cl_init_set_kernel_arg_collisions_search_kernel(){
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
  error = clSetKernelArg(collisions_search_kernel, 3, sizeof(cl_mem), &vx_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/vx_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 4, sizeof(cl_mem), &vy_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/vy_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 5, sizeof(cl_mem), &vz_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/vz_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 6, sizeof(cl_mem), &mass_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/mass_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 7, sizeof(cl_mem), &rad_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/mass_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 8, sizeof(cl_mem), &sort_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/sort_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 9, sizeof(cl_mem), &collisions_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/collisions_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 10, sizeof(cl_mem), &children_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/children_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 11, sizeof(cl_mem), &maxdepth_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/maxdepth_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 12, sizeof(cl_mem), &boxsize_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/boxsize_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 13, sizeof(cl_mem), &num_nodes_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/num_nodes_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 14, sizeof(cl_mem), &num_bodies_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/num_bodies_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 15, sizeof(cl_int)*wave_fronts_in_collisions_search_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/children_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 16, sizeof(cl_int)*wave_fronts_in_collisions_search_kernel*MAX_DEPTH, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/pos_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 17, sizeof(cl_int)*wave_fronts_in_collisions_search_kernel*MAX_DEPTH, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/node_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 18, sizeof(cl_float)*wave_fronts_in_collisions_search_kernel*MAX_DEPTH, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/dr_cutoff_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 19, sizeof(cl_float)*wave_fronts_in_collisions_search_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/nodex_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 20, sizeof(cl_float)*wave_fronts_in_collisions_search_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/nodey_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 21, sizeof(cl_float)*wave_fronts_in_collisions_search_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/nodez_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 22, sizeof(cl_float)*wave_fronts_in_collisions_search_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/nodevx_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 23, sizeof(cl_float)*wave_fronts_in_collisions_search_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/nodevy_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 24, sizeof(cl_float)*wave_fronts_in_collisions_search_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/nodevz_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 25, sizeof(cl_float)*wave_fronts_in_collisions_search_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/noderad_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 26, sizeof(cl_int)*local_size_collisions_search_kernel, NULL);			
  if (error != CL_SUCCESS) {
    fprintf(stderr, "clSetKernelArg Error (collisions_search_kernel/wavefront_vote_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 27, sizeof(cl_mem), &collisions_max2_r_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/collisions_max2_r_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 28, sizeof(cl_mem), &OMEGA_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/OMEGA_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(collisions_search_kernel, 29, sizeof(cl_mem), &t_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (collisions_search_kernel/t_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
}

void cl_init_set_kernel_arg_collisions_resolve_kernel(){
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
}

void cl_init_set_kernel_arg_tree_collisions_kernel(){

//Set Kernel Arguments for tree_collisions_kernel
  error = clSetKernelArg(tree_collisions_kernel, 0, sizeof(cl_mem), &children_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_collisions_kernel/children_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_collisions_kernel, 1, sizeof(cl_mem), &count_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_collisions_kernel/count_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_collisions_kernel, 2, sizeof(cl_mem), &bottom_node_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_collisions_kernel/start_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_collisions_kernel, 3, sizeof(cl_mem), &num_nodes_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_collisions_kernel/sort_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_collisions_kernel, 4, sizeof(cl_mem), &num_bodies_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_collisions_kernel/bottom_node_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }  
  error = clSetKernelArg(tree_collisions_kernel, 5, sizeof(cl_int)*8*local_size_tree_gravity_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_collisions_kernel/children_local): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  } 

}

void cl_init_set_kernel_arg_tree_kernel_no_mass(){

 //Set Kernel Arguments for tree_kernel_no_mass
  error = clSetKernelArg(tree_kernel_no_mass, 0, sizeof(cl_mem), &x_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel_no_mass/x_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel_no_mass, 1, sizeof(cl_mem), &y_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel_no_mass/y_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel_no_mass, 2, sizeof(cl_mem), &z_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel_no_mass/z_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel_no_mass, 3, sizeof(cl_mem), &count_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel_no_mass/count_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel_no_mass, 4, sizeof(cl_mem), &start_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel_no_mass/start_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel_no_mass, 5, sizeof(cl_mem), &children_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel_no_mass/children_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel_no_mass, 6, sizeof(cl_mem), &maxdepth_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel_no_mass/maxdepth_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel_no_mass, 7, sizeof(cl_mem), &bottom_node_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel_no_mass/bottom_node_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel_no_mass, 8, sizeof(cl_mem), &boxsize_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel_no_mass/boxsize_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel_no_mass, 9, sizeof(cl_mem), &rootx_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel_no_mass/rootx_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel_no_mass, 10, sizeof(cl_mem), &rooty_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel_no_mass/rooty_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel_no_mass, 11, sizeof(cl_mem), &rootz_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel_no_mass/rootz_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel_no_mass, 12, sizeof(cl_mem), &num_nodes_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel_no_mass/num_nodes_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel_no_mass, 13, sizeof(cl_mem), &num_bodies_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel_no_mass/num_bodies_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
}

void cl_init_set_kernel_arg_boundaries_kernel(){
 //Set kernel arguments for boundaries kernel
  error = clSetKernelArg(boundaries_kernel, 0, sizeof(cl_mem), &x_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (boundaries_kernel/x_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(boundaries_kernel, 1, sizeof(cl_mem), &y_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (boundaries_kernel/y_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(boundaries_kernel, 2, sizeof(cl_mem), &z_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (boundaries_kernel/z_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(boundaries_kernel, 3, sizeof(cl_mem), &vx_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (boundaries_kernel/vx_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(boundaries_kernel, 4, sizeof(cl_mem), &vy_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (boundaries_kernel/vy_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(boundaries_kernel, 5, sizeof(cl_mem), &vz_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (boundaries_kernel/vz_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(boundaries_kernel, 6, sizeof(cl_mem), &t_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (boundaries_kernel/t_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(boundaries_kernel, 7, sizeof(cl_mem), &boxsize_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (boundaries_kernel/boxsize_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }  
  error = clSetKernelArg(boundaries_kernel, 8, sizeof(cl_mem), &OMEGA_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (boundaries_kernel/OMEGA_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }  
  error = clSetKernelArg(boundaries_kernel, 9, sizeof(cl_mem), &num_bodies_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (boundaries_kernel/num_bodies_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }  
}

void cl_init_set_kernel_arg_integrator_part1_kernel(){
//Set kernel arguments for integrator part 1 kernel
  error = clSetKernelArg(integrator_part1_kernel, 0, sizeof(cl_mem), &x_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part1_kernel/x_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(integrator_part1_kernel, 1, sizeof(cl_mem), &y_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part1_kernel/y_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(integrator_part1_kernel, 2, sizeof(cl_mem), &z_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part1_kernel/z_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(integrator_part1_kernel, 3, sizeof(cl_mem), &vx_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part1_kernel/vx_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(integrator_part1_kernel, 4, sizeof(cl_mem), &vy_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part1_kernel/vy_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(integrator_part1_kernel, 5, sizeof(cl_mem), &vz_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part1_kernel/vz_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(integrator_part1_kernel, 6, sizeof(cl_mem), &ax_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part1_kernel/ax_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(integrator_part1_kernel, 7, sizeof(cl_mem), &ay_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part1_kernel/ay_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(integrator_part1_kernel, 8, sizeof(cl_mem), &az_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part1_kernel/az_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(integrator_part1_kernel, 9, sizeof(cl_mem), &t_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part1_kernel/t_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }  
  error = clSetKernelArg(integrator_part1_kernel, 10, sizeof(cl_mem), &num_bodies_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part1_kernel/num_bodies_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(integrator_part1_kernel, 11, sizeof(cl_mem), &dt_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part1_kernel/dt_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }  
  error = clSetKernelArg(integrator_part1_kernel, 12, sizeof(cl_mem), &OMEGA_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part1_kernel/OMEGA_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  } 
  error = clSetKernelArg(integrator_part1_kernel, 13, sizeof(cl_mem), &OMEGAZ_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part1_kernel/OMEGAZ_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(integrator_part1_kernel, 14, sizeof(cl_mem), &sindt_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part1_kernel/sindt_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }   
  error = clSetKernelArg(integrator_part1_kernel, 15, sizeof(cl_mem), &tandt_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part1_kernel/tandt_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }    
  error = clSetKernelArg(integrator_part1_kernel, 16, sizeof(cl_mem), &sindtz_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part1_kernel/sindtz_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }    
  error = clSetKernelArg(integrator_part1_kernel, 17, sizeof(cl_mem), &tandtz_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part1_kernel/tandtz_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }    
}

void cl_init_set_kernel_arg_integrator_part2_kernel(){

//Set kernel arguments for integrator part 2 kernel
  error = clSetKernelArg(integrator_part2_kernel, 0, sizeof(cl_mem), &x_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part2_kernel/x_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(integrator_part2_kernel, 1, sizeof(cl_mem), &y_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part2_kernel/y_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(integrator_part2_kernel, 2, sizeof(cl_mem), &z_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part2_kernel/z_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(integrator_part2_kernel, 3, sizeof(cl_mem), &vx_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part2_kernel/vx_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(integrator_part2_kernel, 4, sizeof(cl_mem), &vy_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part2_kernel/vy_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(integrator_part2_kernel, 5, sizeof(cl_mem), &vz_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part2_kernel/vz_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(integrator_part2_kernel, 6, sizeof(cl_mem), &ax_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part2_kernel/ax_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(integrator_part2_kernel, 7, sizeof(cl_mem), &ay_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part2_kernel/ay_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(integrator_part2_kernel, 8, sizeof(cl_mem), &az_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part2_kernel/az_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(integrator_part2_kernel, 9, sizeof(cl_mem), &t_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part2_kernel/t_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }  
  error = clSetKernelArg(integrator_part2_kernel, 10, sizeof(cl_mem), &num_bodies_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part2_kernel/num_bodies_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(integrator_part2_kernel, 11, sizeof(cl_mem), &dt_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part2_kernel/dt_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }  
  error = clSetKernelArg(integrator_part2_kernel, 12, sizeof(cl_mem), &OMEGA_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part2_kernel/OMEGA_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  } 
  error = clSetKernelArg(integrator_part2_kernel, 13, sizeof(cl_mem), &OMEGAZ_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part2_kernel/OMEGAZ_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(integrator_part2_kernel, 14, sizeof(cl_mem), &sindt_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part2_kernel/sindt_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }   
  error = clSetKernelArg(integrator_part2_kernel, 15, sizeof(cl_mem), &tandt_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part2_kernel/tandt_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }    
  error = clSetKernelArg(integrator_part2_kernel, 16, sizeof(cl_mem), &sindtz_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part2_kernel/sindtz_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }    
  error = clSetKernelArg(integrator_part2_kernel, 17, sizeof(cl_mem), &tandtz_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (integrator_part2_kernel/tandtz_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }    
}

void cl_init_enqueue_integrator_part1_kernel(){
  error = clEnqueueNDRangeKernel(queue, integrator_part1_kernel, 1, 0, &global_size_integrator_kernel, &local_size_integrator_kernel, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
      fprintf(stderr, "clEqueueNDRangeKernel ERROR (integrator_part1_kernel): %s\n", cl_host_tools_get_error_string(error));
      exit(EXIT_FAILURE);
    }
}

void cl_init_enqueue_boundaries_kernel(){
  error = clEnqueueNDRangeKernel(queue, boundaries_kernel, 1, 0, &global_size_boundaries_kernel, &local_size_boundaries_kernel, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
      fprintf(stderr, "clEqueueNDRangeKernel ERROR (boundaries_kernel): %s\n", cl_host_tools_get_error_string(error));
      exit(EXIT_FAILURE);
    }
}

void cl_init_enqueue_tree_kernel(){
    error = clEnqueueNDRangeKernel(queue, tree_kernel, 1, 0, &global_size_tree_kernel, &local_size_tree_kernel, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
      fprintf(stderr,"clEnqueueNDRangeKernel ERROR (tree_kernel): %s\n",cl_host_tools_get_error_string(error));
      exit(EXIT_FAILURE);
    }
}

void cl_init_enqueue_tree_gravity_kernel(){
   error = clEnqueueNDRangeKernel(queue, tree_gravity_kernel, 1, 0, &global_size_tree_gravity_kernel, &local_size_tree_gravity_kernel, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
      fprintf(stderr,"clEnqueueNDRangeKernel ERROR (tree_gravity_kernel): %s\n",cl_host_tools_get_error_string(error));
      exit(EXIT_FAILURE);
    }
}

void cl_init_enqueue_tree_sort_kernel(){
    error = clEnqueueNDRangeKernel(queue, tree_sort_kernel, 1, 0, &global_size_tree_sort_kernel, &local_size_tree_sort_kernel, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
      fprintf(stderr,"clEnqueueNDRangeKernel ERROR (tree_sort_kernel): %s\n",cl_host_tools_get_error_string(error));
      exit(EXIT_FAILURE);
    }
}

void cl_init_enqueue_force_gravity_kernel(){
    error = clEnqueueNDRangeKernel(queue, force_gravity_kernel, 1, 0, &global_size_force_gravity_kernel, &local_size_force_gravity_kernel, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
      fprintf(stderr, "clEqueueNDRangeKernel ERROR (force_gravity_kernel): %s\n", cl_host_tools_get_error_string(error));
      exit(EXIT_FAILURE);
    }
}

void cl_init_enqueue_integrator_part2_kernel(){
    error = clEnqueueNDRangeKernel(queue, integrator_part2_kernel, 1, 0, &global_size_integrator_kernel, &local_size_integrator_kernel, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
      fprintf(stderr, "clEqueueNDRangeKernel ERROR (integrator_part2_kernel): %s\n", cl_host_tools_get_error_string(error));
      exit(EXIT_FAILURE);
    }
}

void cl_init_enqueue_tree_kernel_no_mass(){
   error = clEnqueueNDRangeKernel(queue, tree_kernel_no_mass, 1, 0, &global_size_tree_kernel, &local_size_tree_kernel, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
      fprintf(stderr,"clEnqueueNDRangeKernel ERROR (tree_kernel): %s\n",cl_host_tools_get_error_string(error));
      exit(EXIT_FAILURE);
    }
}

void cl_init_enqueue_tree_collisions_kernel(){
   error = clEnqueueNDRangeKernel(queue, tree_collisions_kernel, 1, 0, &global_size_tree_gravity_kernel, &local_size_tree_gravity_kernel, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
      fprintf(stderr,"clEnqueueNDRangeKernel ERROR (tree_gravity_kernel): %s\n",cl_host_tools_get_error_string(error));
      exit(EXIT_FAILURE);
    }
}

void cl_init_enqueue_collisions_search_kernel(){
    error = clEnqueueNDRangeKernel(queue, collisions_search_kernel, 1, 0, &global_size_collisions_search_kernel, &local_size_collisions_search_kernel, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
      fprintf(stderr, "clEqueueNDRangeKernel ERROR (collisions_search_kernel): %s\n", cl_host_tools_get_error_string(error));
      exit(EXIT_FAILURE);
    }
}

void cl_init_enqueue_collisions_resolve_kernel(){
  error = clEnqueueNDRangeKernel(queue, collisions_resolve_kernel, 1, 0, &global_size_collisions_resolve_kernel, &local_size_collisions_resolve_kernel, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
      fprintf(stderr, "clEqueueNDRangeKernel ERROR (collisions_resolve_kernel): %s\n", cl_host_tools_get_error_string(error));
      exit(EXIT_FAILURE);
    }
}

void cl_init_free_globals(){

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
  clReleaseMemObject(G_buffer);
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
  clReleaseMemObject(dt_buffer);
  clReleaseMemObject(t_buffer);
  clReleaseMemObject(rad_buffer);
  clReleaseMemObject(collisions_max2_r_buffer);
  clReleaseMemObject(OMEGA_buffer);
  clReleaseMemObject(OMEGAZ_buffer);
  clReleaseMemObject(sindt_buffer);
  clReleaseMemObject(tandt_buffer);
  clReleaseMemObject(sindtz_buffer);
  clReleaseMemObject(tandtz_buffer);

  clReleaseKernel(tree_kernel);
  clReleaseKernel(tree_kernel_no_mass);
  clReleaseKernel(tree_gravity_kernel);
  clReleaseKernel(tree_collisions_kernel);
  clReleaseKernel(tree_sort_kernel);
  clReleaseKernel(force_gravity_kernel);
  clReleaseKernel(collisions_search_kernel);
  clReleaseKernel(collisions_resolve_kernel);
  clReleaseKernel(boundaries_kernel);
  clReleaseKernel(integrator_part1_kernel);
  clReleaseKernel(integrator_part2_kernel);

  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);

}
