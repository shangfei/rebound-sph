#define WARP_SIZE 32

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "../../src/cl_host_tools.h"
#include "tree_cl_test.h"


void tree_cl_test(
		     float *x_host,
		     float *y_host,
		     float *z_host,
		     float *boxsize,
		     float *rootx_host,
		     float *rooty_host,
		     float *rootz_host,
		     int *num_bodies_host
		  )
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

  cl_int num_nodes_host;

  device = cl_host_tools_create_device();
  
  work_groups = cl_host_tools_get_num_compute_units(device);
  local_size = 128;
  global_size = work_groups*local_size;
  
  num_nodes_host = *num_bodies_host * 2;
  if (num_nodes_host < 1024*work_groups) 
    num_nodes_host = 1024*work_groups;
  while (num_nodes_host % WARP_SIZE != 0) num_nodes_host ++;

  //we will be using num_nodes to retrieve the last array element, so we must decrement.
  num_nodes_host--; 

  x_host = (float *) realloc (x_host, (num_nodes_host + 1) * sizeof(float));
  y_host = (float *) realloc (y_host, (num_nodes_host + 1) * sizeof(float));
  z_host = (float *) realloc (z_host, (num_nodes_host + 1) * sizeof(float));

  printf(" work items per workgroup = %u\n", (unsigned int)local_size);
  printf(" work groups = %u\n", (unsigned int)work_groups);
  printf(" total work items = %u\n", (unsigned int)global_size);

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
  if (error < 0) {
    perror("Couldn't create a CL context");
    exit(EXIT_FAILURE);
  }

  program = cl_host_tools_create_program(context, device, "../../src/cl_tree.cl");
  
  // Create buffers
  x_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (num_nodes_host+1) * sizeof(cl_float), x_host, &error);
  y_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (num_nodes_host+1) * sizeof(cl_float), y_host, &error);
  z_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (num_nodes_host+1) * sizeof(cl_float), z_host, &error);
  
  mass_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, (num_nodes_host+1) * sizeof(cl_float), NULL, &error);
  children_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 8 * (num_nodes_host+1) * sizeof(cl_int), NULL, &error);
  bottom_node_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &error); 

  num_nodes_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_int), &num_nodes_host, &error);
  num_bodies_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_int), num_bodies_host,  &error);
  rootx_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), rootx_host, &error);
  rooty_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), rooty_host, &error);
  rootz_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float), rootz_host, &error);

  if (error < 0) {
    perror("Couldn't create a CL buffer");
    exit(EXIT_FAILURE);
  }

  //Create a command queue
  queue = clCreateCommandQueue(context, device, 0, &error);
  if (error < 0){
    perror("Couldn't create a command queue");
    exit(EXIT_FAILURE);
  };

  //Create kernels
  tree_kernel = clCreateKernel(program, "cl_build_tree", &error);
  if (error < 0){
    perror("Couldn't create a kernel");
    exit(EXIT_FAILURE);
  }

  error = clSetKernelArg(tree_kernel, 0, sizeof(cl_mem), &x_buffer);
  error |= clSetKernelArg(tree_kernel, 1, sizeof(cl_mem), &y_buffer);
  error |= clSetKernelArg(tree_kernel, 2, sizeof(cl_mem), &z_buffer);
  error |= clSetKernelArg(tree_kernel, 3, sizeof(cl_mem), &mass_buffer);
  error |= clSetKernelArg(tree_kernel, 4, sizeof(cl_mem), &children_buffer);
  error |= clSetKernelArg(tree_kernel, 5, sizeof(cl_mem), &bottom_node_buffer);
  error |= clSetKernelArg(tree_kernel, 6, sizeof(cl_mem), &boxsize_buffer);
  error |= clSetKernelArg(tree_kernel, 7, sizeof(cl_mem), &rootx_buffer);
  error |= clSetKernelArg(tree_kernel, 8, sizeof(cl_mem), &rooty_buffer);
  error |= clSetKernelArg(tree_kernel, 9, sizeof(cl_mem), &rootz_buffer);
  error |= clSetKernelArg(tree_kernel, 10, sizeof(cl_mem), &num_nodes_buffer);
  error |= clSetKernelArg(tree_kernel, 11, sizeof(cl_mem), &num_bodies_buffer);

  if (error < 0) {
    perror("clSetKernelArg failed");
    exit(EXIT_FAILURE);
  }

  // Enqueue kernel
  error = clEnqueueNDRangeKernel(queue, tree_kernel, 1, 0, &global_size, &local_size, 0, NULL, NULL);
  
  clFinish(queue);

  // Read the result 
  int *children_host;
  int *bottom_node_host; 
  children_host = (int *) malloc( sizeof(int) * (num_nodes_host + 1) * 8);
  bottom_node_host = (int *) malloc( sizeof(int) );

  error = clEnqueueReadBuffer(queue, children_buffer, CL_TRUE, 0, sizeof(int) * 8 * (num_nodes_host + 1), children_host, 0, NULL, NULL);
  error |= clEnqueueReadBuffer(queue, bottom_node_buffer, CL_TRUE, 0, sizeof(int), bottom_node_host, 0, NULL, NULL);

  if(error < 0) {
    perror("Couldn't read the buffer");
    exit(1);   
  }
  
  for (int node = *bottom_node_host; node < num_nodes_host + 1; node++){
    printf("\n+++++NODE %d+++++:",node);
    for (int child = 0; child < 8; child++)
      printf(" %d ", children_host[node*8 + child]);
  }

  free(children_host);
  free(bottom_node_host);

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
