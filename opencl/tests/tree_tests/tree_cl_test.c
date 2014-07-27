#define WARP_SIZE 32

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "../../src/cl_host_tools.h"
#include "tree_cl_test.h"


void tree_cl_test()
{
  cl_device_id device;
  cl_context context;
  cl_program program;
  cl_kernel tree_kernel;
  cl_command_queue queue;
  size_t local_size, global_size;
  cl_int error;
  cl_int work_groups;
  cl_mem x_buffer;
  cl_mem y_buffer;
  cl_mem z_buffer;
  cl_mem mass_buffer;
  cl_mem children_buffer;
  cl_mem bottom_node_buffer;
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
  cl_int bottom_node_host;
  cl_int num_nodes_host;
  cl_int num_bodies_host = 32768;
  cl_float boxsize_host = 10;
  cl_float rootx_host = 0.f;
  cl_float rooty_host = 0.f;
  cl_float rootz_host = 0.f;

  device = cl_host_tools_create_device();
  
  work_groups = cl_host_tools_get_num_compute_units(device);
  local_size = 128;
  global_size = work_groups*local_size;
 
  //  each leaf belongs to one parent node, so we at least need space for num_bodies of nodes
  //and we will need space for num_bodies of bodies, so:
  num_nodes_host = num_bodies_host * 2;
  if (num_nodes_host < 1024*work_groups)
    num_nodes_host = 1024*work_groups;
  while (num_nodes_host % WARP_SIZE != 0) (num_nodes_host) ++;

  //we will be using num_nodes to retrieve the last array element, so we must decrement.
  (num_nodes_host)--;

  children_host = (cl_int *) malloc( sizeof(cl_int) * ((num_nodes_host) + 1) * 8);

  x_host = (cl_float *) malloc ( ((num_nodes_host) + 1) * sizeof(cl_float));
  y_host = (cl_float *) malloc ( ((num_nodes_host) + 1) * sizeof(cl_float));
  z_host = (cl_float *) malloc ( ((num_nodes_host) + 1) * sizeof(cl_float));
  
  srand(time(NULL));
  for (int i = 0; i < num_bodies_host; i++){
    x_host[i] = rootx_host - boxsize_host/2.f + boxsize_host*cl_host_tools_random_float();
    y_host[i] = rooty_host - boxsize_host/2.f + boxsize_host*cl_host_tools_random_float();
    z_host[i] = rootz_host - boxsize_host/2.f + boxsize_host*cl_host_tools_random_float();
  }
	 
  printf(" work items per workgroup = %u\n", (unsigned int)local_size);
  printf(" work groups = %u\n", (unsigned int)work_groups);
  printf(" total work items = %u\n", (unsigned int)global_size);
  printf(" num_nodes_host = %d\n", num_nodes_host);
  printf(" num_bodies_host = %d\n",num_bodies_host);


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
  mass_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, (num_nodes_host+1) * sizeof(cl_float), NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clCreateBuffer ERROR (mass_buffer): %d\n",error);
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
  tree_kernel = clCreateKernel(program, "cl_build_tree", &error);
  if (error != CL_SUCCESS){
    fprintf(stderr,"clCreateKernel ERROR: %d\n",error);
    exit(EXIT_FAILURE);
  }

  error = clSetKernelArg(tree_kernel, 0, sizeof(cl_mem), &x_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (x_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 1, sizeof(cl_mem), &y_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (y_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 2, sizeof(cl_mem), &z_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (z_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 3, sizeof(cl_mem), &mass_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (mass_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 4, sizeof(cl_mem), &children_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (children_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 5, sizeof(cl_mem), &bottom_node_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (bottom_node_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 6, sizeof(cl_mem), &boxsize_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (boxsize_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 7, sizeof(cl_mem), &rootx_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (rootx_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 8, sizeof(cl_mem), &rooty_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (rooty_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 9, sizeof(cl_mem), &rootz_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (rootz_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 10, sizeof(cl_mem), &num_nodes_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (num_nodes_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }
  error = clSetKernelArg(tree_kernel, 11, sizeof(cl_mem), &num_bodies_buffer);
  if (error != CL_SUCCESS) {
    fprintf(stderr,"clSetKernelArg ERROR (num_bodies_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }

  //Enqueue kernel
  error = clEnqueueNDRangeKernel(queue, tree_kernel, 1, 0, &global_size, &local_size, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueNDRangeKernel ERROR: %d\n",error);
    exit(EXIT_FAILURE);
  }

  // Read the results
  error = clEnqueueReadBuffer(queue, children_buffer, CL_TRUE, 0, sizeof(cl_int) * 8 * (num_nodes_host + 1), children_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (children_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }

  error |= clEnqueueReadBuffer(queue, bottom_node_buffer, CL_TRUE, 0, sizeof(cl_int), &bottom_node_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (bottom_node_buffer): %d\n",error);
    exit(EXIT_FAILURE);
  }

  printf("\n bottom_node_host = %d\n", (int)bottom_node_host);

  //Block until queue is done
  error = clFinish(queue);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clFinish ERROR: %d\n",error);
    exit(EXIT_FAILURE);
  }

  for (int node = bottom_node_host; node < num_nodes_host + 1; node++){
    printf("+++++NODE %d+++++:",node);
    for (int child = 0; child < 8; child++)
      printf(" %d ", children_host[node*8 + child]);
    printf("\n");
  }


  free(children_host);
  free(x_host);
  free(y_host);
  free(z_host);

  clReleaseMemObject(x_buffer);
  clReleaseMemObject(y_buffer);
  clReleaseMemObject(z_buffer);
  clReleaseMemObject(mass_buffer);
  clReleaseMemObject(children_buffer);
  clReleaseMemObject(bottom_node_buffer);
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
