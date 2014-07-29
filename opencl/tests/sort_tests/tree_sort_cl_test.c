#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "cl_gpu_defns.h"
#include "../../src/cl_host_tools.h"
#include "tree_sort_cl_test.h"

void tree_sort_cl_test(int num_bodies, int num_threads_tree_kernel, int num_threads_tree_gravity_kernel, int num_threads_tree_sort_kernel)
{
  cl_device_id device;
  cl_context context;
  cl_program program;
  cl_kernel tree_kernel;
  cl_kernel tree_gravity_kernel;
  cl_kernel tree_sort_kernel;
  cl_command_queue queue;
  size_t local_size_tree_kernel;
  size_t global_size_tree_kernel;
  size_t local_size_tree_gravity_kernel;
  size_t global_size_tree_gravity_kernel;
  size_t local_size_tree_sort_kernel;
  size_t global_size_tree_sort_kernel;
  cl_int error;
  cl_int work_groups;
  cl_mem x_buffer;
  cl_mem y_buffer;
  cl_mem z_buffer;
  cl_mem mass_buffer;
  cl_mem start_buffer;
  cl_mem sort_buffer;
  cl_mem count_buffer;
  cl_mem children_buffer;
  cl_mem bottom_node_buffer;
  cl_mem maxdepth_buffer;
  cl_mem boxsize_buffer;
  cl_mem rootx_buffer;
  cl_mem rooty_buffer;
  cl_mem rootz_buffer;
  cl_mem num_nodes_buffer;
  cl_mem num_bodies_buffer;
  cl_int *children_host;
  cl_float *x_host;
  cl_float *y_host;
  cl_float *z_host;
  cl_float *mass_host;
  cl_int *count_host;
  cl_int *sort_host;
  cl_int *start_host;
  cl_int bottom_node_host;
  cl_int maxdepth_host;
  cl_int num_nodes_host;
  cl_int num_bodies_host = num_bodies;
  cl_float boxsize_host = 10;
  cl_float rootx_host = 0.f;
  cl_float rooty_host = 0.f;
  cl_float rootz_host = 0.f;

  device = cl_host_tools_create_device();
  
  work_groups = cl_host_tools_get_num_compute_units(device);

  local_size_tree_kernel = num_threads_tree_kernel;
  global_size_tree_kernel = work_groups*local_size_tree_kernel;

  local_size_tree_gravity_kernel = num_threads_tree_gravity_kernel;
  global_size_tree_gravity_kernel = work_groups*local_size_tree_gravity_kernel;

  local_size_tree_sort_kernel = num_threads_tree_sort_kernel;
  global_size_tree_sort_kernel = work_groups*local_size_tree_sort_kernel;
 
  //each leaf belongs to one parent node, so we at least need space for num_bodies of nodes
  //and we will need space for num_bodies of bodies, so:
  num_nodes_host = num_bodies_host * 2;
  if (num_nodes_host < 1024*work_groups)
    num_nodes_host = 1024*work_groups;
  while (num_nodes_host % WARP_SIZE != 0) (num_nodes_host) ++;

  //we will be using num_nodes to retrieve the last array element, so we must decrement.
  (num_nodes_host)--;

  children_host = (cl_int *) malloc( sizeof(cl_int) * ((num_nodes_host) + 1) * 8);
  
  x_host = (cl_float *) malloc ( (num_nodes_host + 1) * sizeof(cl_float));
  y_host = (cl_float *) malloc ( (num_nodes_host + 1) * sizeof(cl_float));
  z_host = (cl_float *) malloc ( (num_nodes_host + 1) * sizeof(cl_float));
  mass_host = (cl_float *) malloc ( (num_nodes_host + 1) * sizeof(cl_float));
  count_host = (cl_int *) malloc ( (num_nodes_host + 1) * sizeof(cl_int));
  sort_host = (cl_int *) malloc ( (num_nodes_host + 1) * sizeof(cl_int));
  start_host = (cl_int *) malloc ( (num_nodes_host + 1) * sizeof(cl_int));

  srand(time(NULL));
  for (cl_int i = 0; i < num_bodies_host; i++){
    x_host[i] = rootx_host - boxsize_host/2.f + boxsize_host*cl_host_tools_random_float();
    y_host[i] = rooty_host - boxsize_host/2.f + boxsize_host*cl_host_tools_random_float();
    z_host[i] = rootz_host - boxsize_host/2.f + boxsize_host*cl_host_tools_random_float();
    mass_host[i] = 1.0f/num_bodies_host;
    printf( "%d %.6f %.6f %.6f\n", i, x_host[i], y_host[i], z_host[i]);
  }
	 
  printf(" work items per workgroup in tree kernel = %u\n", (unsigned int)local_size_tree_kernel);
  printf(" work items per workgroup in tree gravity kernel = %u\n", (unsigned int)local_size_tree_gravity_kernel);
  printf(" work groups = %u\n", (unsigned int)work_groups);
  printf(" total work items in tree kernel = %u\n", (unsigned int)global_size_tree_kernel);
  printf(" total work items in tree gravity kernel = %u\n", (unsigned int)global_size_tree_gravity_kernel);
  printf(" num_nodes_host = %d\n", num_nodes_host);
  printf(" num_bodies_host = %d\n", num_bodies_host);

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateContext ERROR");
    exit(EXIT_FAILURE);
  }

  program = cl_host_tools_create_program(context, device, "../../src/cl_tree.cl");
  
  // Create buffers
  x_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (num_nodes_host+1) * sizeof(cl_float), x_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (x_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  y_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (num_nodes_host+1) * sizeof(cl_float), y_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (y_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  z_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (num_nodes_host+1) * sizeof(cl_float), z_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (z_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  mass_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (num_nodes_host+1) * sizeof(cl_float), mass_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (mass_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }

  start_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, (num_nodes_host+1) * sizeof(cl_int), NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (start_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }

  sort_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, (num_nodes_host+1) * sizeof(cl_int), NULL, &error);
  if (error != CL_SUCCESS){
    fprintf(stderr, "clCreateBuffer ERROR (sort_buffer): %d\n", error);
    exit(EXIT_FAILURE);
  }

  count_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, (num_nodes_host+1) * sizeof(cl_int), NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr, "clCreateBuffer ERROR (count_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }

  children_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 8 * (num_nodes_host+1) * sizeof(cl_int), NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (children_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  bottom_node_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (bottom_node_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  maxdepth_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (maxdepth_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  num_nodes_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_int), &num_nodes_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (num_nodes_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  num_bodies_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_int), &num_bodies_host,  &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (num_bodies_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  boxsize_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &boxsize_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (boxsize_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  rootx_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &rootx_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (rootx_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  rooty_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &rooty_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (rooty_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  rootz_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), &rootz_host, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (rootz_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }

  //Create a command queue
  queue = clCreateCommandQueue(context, device, 0, &error);
  if (error != CL_SUCCESS){
    fprintf(stderr,"clCreateCommandQueue ERROR: %d\n",error);
    exit(EXIT_FAILURE);
  };

  //Create kernels
  tree_kernel = clCreateKernel(program, "cl_tree_add_particles_to_tree", &error);
  if (error != CL_SUCCESS){
    fprintf(stderr,"clCreateKernel ERROR (tree_kernel): %d\n",error);
    exit(EXIT_FAILURE);
  }

  tree_gravity_kernel = clCreateKernel(program, "cl_tree_update_tree_gravity_data", &error);
  if (error != CL_SUCCESS){
    fprintf(stderr, "clCreateKernel ERROR (tree_gravity_kernel): %d\n", error);
    exit(EXIT_FAILURE);
  }

  tree_sort_kernel = clCreateKernel(program, "cl_tree_sort_particles", &error);
  if (error != CL_SUCCESS){
    fprintf(stderr, "clCreateKernel ERROR (tree_sort_kernel): %d\n", error);
    exit(EXIT_FAILURE);
  }

  //Set Kernel Arguments for tree_sort_kernel
  error = clSetKernelArg(tree_sort_kernel, 0, sizeof(cl_mem), &children_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_sort_kernel/children_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_sort_kernel, 1, sizeof(cl_mem), &count_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_sort_kernel/count_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_sort_kernel, 2, sizeof(cl_mem), &start_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_sort_kernel/start_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_sort_kernel, 3, sizeof(cl_mem), &sort_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_sort_kernel/sort_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_sort_kernel, 4, sizeof(cl_mem), &bottom_node_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_sort_kernel/bottom_node_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }  
  error = clSetKernelArg(tree_sort_kernel, 5, sizeof(cl_mem), &num_nodes_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_sort_kernel/num_nodes_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_sort_kernel, 6, sizeof(cl_mem), &num_bodies_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_sort_kernel/num_bodies_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }

  //Set Kernel Arguments for tree_gravity_kernel
  error = clSetKernelArg(tree_gravity_kernel, 0, sizeof(cl_mem), &x_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/x_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 1, sizeof(cl_mem), &y_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/y_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 2, sizeof(cl_mem), &z_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/z_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 3, sizeof(cl_mem), &mass_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/mass_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 4, sizeof(cl_mem), &children_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/children_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 5, sizeof(cl_mem), &count_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/count_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 6, sizeof(cl_mem), &bottom_node_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/bottom_node_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }  
  error = clSetKernelArg(tree_gravity_kernel, 7, sizeof(cl_mem), &num_nodes_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/num_nodes_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 8, sizeof(cl_mem), &num_bodies_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/num_bodies_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_gravity_kernel, 9, sizeof(cl_int)*8*num_threads_tree_gravity_kernel, NULL);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_gravity_kernel/children_local): %d\n",error);
    exit(EXIT_FAILURE);
  } 
  
  //Set Kernel Arguments for tree_kernel  
  error = clSetKernelArg(tree_kernel, 0, sizeof(cl_mem), &x_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/x_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 1, sizeof(cl_mem), &y_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/y_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 2, sizeof(cl_mem), &z_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/z_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 3, sizeof(cl_mem), &mass_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/mass_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 4, sizeof(cl_mem), &start_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/start_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 5, sizeof(cl_mem), &children_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/children_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 6, sizeof(cl_mem), &maxdepth_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/maxdepth_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 7, sizeof(cl_mem), &bottom_node_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/bottom_node_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 8, sizeof(cl_mem), &boxsize_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/boxsize_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 9, sizeof(cl_mem), &rootx_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/rootx_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 10, sizeof(cl_mem), &rooty_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/rooty_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 11, sizeof(cl_mem), &rootz_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/rootz_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 12, sizeof(cl_mem), &num_nodes_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/num_nodes_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 13, sizeof(cl_mem), &num_bodies_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (tree_kernel/num_bodies_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }

  //Enqueue kernels
  error = clEnqueueNDRangeKernel(queue, tree_kernel, 1, 0, &global_size_tree_kernel, &local_size_tree_kernel, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueNDRangeKernel ERROR (tree_kernel): %d\n",error);
    exit(EXIT_FAILURE);
  }

  error = clEnqueueNDRangeKernel(queue, tree_gravity_kernel, 1, 0, &global_size_tree_gravity_kernel, &local_size_tree_gravity_kernel, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueNDRangeKernel ERROR (tree_gravity_kernel): %d\n",error);
    exit(EXIT_FAILURE);
  }

 error = clEnqueueNDRangeKernel(queue, tree_sort_kernel, 1, 0, &global_size_tree_sort_kernel, &local_size_tree_sort_kernel, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueNDRangeKernel ERROR (tree_sort_kernel): %d\n",error);
    exit(EXIT_FAILURE);
  }

  //Block until queue is done
  error = clFinish(queue);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clFinish ERROR: %d\n",error);
    exit(EXIT_FAILURE);
  }

  // Read the results
  error = clEnqueueReadBuffer(queue, children_buffer, CL_TRUE, 0, sizeof(cl_int) * 8 * (num_nodes_host + 1), children_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (children_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, bottom_node_buffer, CL_TRUE, 0, sizeof(cl_int), &bottom_node_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (bottom_node_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, maxdepth_buffer, CL_TRUE, 0, sizeof(cl_int), &maxdepth_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (maxdepth_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, mass_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_nodes_host + 1), mass_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (mass_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, start_buffer, CL_TRUE, 0, sizeof(cl_int) * (num_nodes_host + 1), start_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (start_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, sort_buffer, CL_TRUE, 0, sizeof(cl_int) * (num_nodes_host + 1), sort_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (sort_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, x_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_nodes_host + 1), x_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (x_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, y_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_nodes_host + 1), y_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (y_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, z_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_nodes_host + 1), z_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (z_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, count_buffer, CL_TRUE, 0, sizeof(cl_int) * (num_nodes_host + 1), count_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (count_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }

  printf("\n++++++TREE++++++\n");

  for (int node = bottom_node_host; node < num_nodes_host + 1; node++){
    printf("+++++NODE %d+++++:",node);
    for (int child = 0; child < 8; child++)
      printf(" %d ", children_host[node*8 + child]);
    printf("\n");
  }

  printf("\n++++++XYZ and MASS++++++\n");
  
  for (int i = 0; i < num_nodes_host + 1; i++){
    printf(" %d %.6f %.6f %.6f %.6f\n", i, x_host[i], y_host[i], z_host[i], mass_host[i]);
  }

  printf("\n++++++COUNT_HOST++++++\n");
  for (int i = 0; i < num_nodes_host + 1; i++){
    printf(" %d %d ", i, count_host[i]);
  }

  printf("\n++++++SORT_HOST++++++\n");
  for (int i = 0; i < num_nodes_host + 1; i++){
    printf(" %d %d ", i, sort_host[i]);
  }

  printf("\n++++++START_HOST++++++\n"); 
  for (int i = 0; i < num_nodes_host + 1; i++){
    printf(" %d %d ", i, start_host[i]);
  }

  printf("\n bottom_node_host = %d\n", (int)bottom_node_host);
  printf("maxdepth_host = %d\n", (int)maxdepth_host);

  free(children_host);
  free(mass_host);
  free(count_host);
  free(start_host);
  free(sort_host);
  free(x_host);
  free(y_host);
  free(z_host);

  clReleaseMemObject(x_buffer);
  clReleaseMemObject(y_buffer);
  clReleaseMemObject(z_buffer);
  clReleaseMemObject(mass_buffer);
  clReleaseMemObject(start_buffer);
  clReleaseMemObject(sort_buffer);
  clReleaseMemObject(count_buffer);
  clReleaseMemObject(children_buffer);
  clReleaseMemObject(bottom_node_buffer);
  clReleaseMemObject(maxdepth_buffer);
  clReleaseMemObject(boxsize_buffer);
  clReleaseMemObject(num_nodes_buffer);
  clReleaseMemObject(num_bodies_buffer);
  clReleaseMemObject(rootx_buffer);
  clReleaseMemObject(rooty_buffer);
  clReleaseMemObject(rootz_buffer);
  clReleaseKernel(tree_kernel);
  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);

}
