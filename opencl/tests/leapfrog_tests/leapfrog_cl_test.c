#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "../../src/cl_host_tools.h"
#include "leapfrog_cl_test.h"
#include "leapfrog_params.h"

#define PROGRAM_FILE "../../src/cl_integrator_leapfrog.cl"
#define KERNEL_1 "cl_integrator_leapfrog_part1"
#define KERNEL_2 "cl_integrator_leapfrog_part2"


void leapfrog_cl_test(float x [], float y [], float z [], float vx [], float vy [], float vz [], float dt)
{
  cl_device_id device;
  cl_context context;
  cl_program program;
  cl_kernel leapfrog_part1_kernel;
  cl_kernel leapfrog_part2_kernel;
  cl_command_queue queue;
  size_t local_size, global_size;
  cl_int error;
  cl_int work_groups;

  cl_mem x_buffer;
  cl_mem y_buffer;
  cl_mem z_buffer;
  cl_mem vx_buffer;
  cl_mem vy_buffer;
  cl_mem vz_buffer; 

  device = cl_host_tools_create_device();
  error = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(local_size), &local_size, NULL);
  local_size = 128;
  work_groups = (int) ( ((float)N) / ((float)local_size) );
  global_size = work_groups*local_size;

  printf(" work items per workgroup = %u\n", (unsigned int)local_size);
  printf(" work groups = %u\n", (unsigned int)work_groups);
  printf(" total work items = %u\n", (unsigned int)global_size);

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
  if (error < 0) {
    perror("Couldn't create a CL context");
    exit(EXIT_FAILURE);
  }

  program = cl_host_tools_create_program(context, device, PROGRAM_FILE);
  
  // Create buffers
  x_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * sizeof(float), x, &error);
  y_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * sizeof(float), y, &error);
  z_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * sizeof(float), z, &error);
  vx_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * sizeof(float), vx, &error);
  vy_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * sizeof(float), vy, &error);
  vz_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * sizeof(float), vz, &error);
 
  if (error < 0) {
    perror("Couldn't create a CL buffer");
    exit(EXIT_FAILURE);
  }

  //Create a command queue
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
  if (error < 0){
    perror("Couldn't create a command queue");
    exit(EXIT_FAILURE);
  };

  //Create kernels
  leapfrog_part1_kernel = clCreateKernel(program, KERNEL_1, &error);
  leapfrog_part2_kernel = clCreateKernel(program, KERNEL_2, &error);
  if (error < 0){
    perror("Couldn't create a kernel");
    exit(EXIT_FAILURE);
  }

  error = clSetKernelArg(leapfrog_part1_kernel, 0, sizeof(cl_mem), &x_buffer);
  error |= clSetKernelArg(leapfrog_part1_kernel, 1, sizeof(cl_mem), &y_buffer);
  error |= clSetKernelArg(leapfrog_part1_kernel, 2, sizeof(cl_mem), &z_buffer);
  error |= clSetKernelArg(leapfrog_part1_kernel, 3, sizeof(cl_mem), &vx_buffer);
  error |= clSetKernelArg(leapfrog_part1_kernel, 4, sizeof(cl_mem), &vy_buffer);
  error |= clSetKernelArg(leapfrog_part1_kernel, 5, sizeof(cl_mem), &vz_buffer);
  error |= clSetKernelArg(leapfrog_part1_kernel, 6, sizeof(dt), &dt);

  error = clSetKernelArg(leapfrog_part2_kernel, 0, sizeof(cl_mem), &x_buffer);
  error |= clSetKernelArg(leapfrog_part2_kernel, 1, sizeof(cl_mem), &y_buffer);
  error |= clSetKernelArg(leapfrog_part2_kernel, 2, sizeof(cl_mem), &z_buffer);
  error |= clSetKernelArg(leapfrog_part2_kernel, 3, sizeof(cl_mem), &vx_buffer);
  error |= clSetKernelArg(leapfrog_part2_kernel, 4, sizeof(cl_mem), &vy_buffer);
  error |= clSetKernelArg(leapfrog_part2_kernel, 5, sizeof(cl_mem), &vz_buffer);
  error |= clSetKernelArg(leapfrog_part2_kernel, 6, sizeof(dt), &dt);

  if (error < 0) {
    perror("Couldn't create a kernel argument");
    exit(EXIT_FAILURE);
  }

  // Enqueue kernels
  for (int j = 0; j < RUNS; j++){
    error = clEnqueueNDRangeKernel(queue, leapfrog_part1_kernel, 1, 0, &global_size, &local_size, 0, NULL, NULL);
    error |= clEnqueueNDRangeKernel(queue, leapfrog_part2_kernel, 1, 0, &global_size, &local_size,0, NULL, NULL);
  }
  // Finish processing the queue and get profiling information
  clFinish(queue);

  // Read the result 
  error = clEnqueueReadBuffer(queue, x_buffer, CL_TRUE, 0, sizeof(float) * N, x, 0, NULL, NULL);
  error |= clEnqueueReadBuffer(queue, y_buffer, CL_TRUE, 0, sizeof(float) * N, y, 0, NULL, NULL);
  error |= clEnqueueReadBuffer(queue, z_buffer, CL_TRUE, 0, sizeof(float) * N, z, 0, NULL, NULL);
  error |= clEnqueueReadBuffer(queue, vx_buffer, CL_TRUE, 0, sizeof(float) * N, vx, 0, NULL, NULL);
  error |= clEnqueueReadBuffer(queue, vy_buffer, CL_TRUE, 0, sizeof(float) * N, vy, 0, NULL, NULL);
  error |= clEnqueueReadBuffer(queue, vz_buffer, CL_TRUE, 0, sizeof(float) * N, vz, 0, NULL, NULL);
  if(error < 0) {
    perror("Couldn't read the buffer");
    exit(1);   
  }

  /* Deallocate resources */
  clReleaseMemObject(x_buffer);
  clReleaseMemObject(y_buffer);
  clReleaseMemObject(z_buffer);
  clReleaseMemObject(vx_buffer);
  clReleaseMemObject(vy_buffer);
  clReleaseMemObject(vz_buffer);
  clReleaseKernel(leapfrog_part1_kernel);
  clReleaseKernel(leapfrog_part2_kernel);
  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);

}
