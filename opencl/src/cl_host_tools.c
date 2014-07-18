#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

float cl_host_tools_random_float(){
  return (float)rand() / (float)RAND_MAX;
}

cl_device_id cl_host_tools_create_device(){

  cl_platform_id platform;
  cl_device_id device;
  int error;
 
  //identify the first platform
  error = clGetPlatformIDs(1, &platform, NULL);
  if (error < 0){
    perror("Couldn't find a platform");
    exit(EXIT_FAILURE);
  }

  //get first GPU device available
  error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  if(error == CL_DEVICE_NOT_FOUND){
    printf("Could not find a GPU device");
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  }
  if(error < 0) {
    perror("Couldn't access any device");
    exit(EXIT_FAILURE);
  }
  return device;

}

cl_program cl_host_tools_create_program(cl_context context, cl_device_id device, const char* filename){

  cl_program program;
  FILE *program_handle;
  char *program_buffer;
  char *program_log;
  int program_size;
  int log_size;
  int error;

  //read in program
  program_handle = fopen(filename, "r");
  if(program_handle == NULL) {
    perror("Couldn't find the program file");
    exit(EXIT_FAILURE);
  }
  fseek(program_handle, 0, SEEK_END);
  program_size = ftell(program_handle);
  rewind(program_handle);
  program_buffer = (char*)malloc(program_size + 1);
  program_buffer[program_size] = '\0';
  fread(program_buffer,sizeof(char), program_size, program_handle);
  fclose(program_handle);

  //Create program from file
  program = clCreateProgramWithSource(context, 1, (const char**)&program_buffer, &program_size, &error);
  if(error < 0){
    perror("Couldn't create the program");
    exit(EXIT_FAILURE);
  }
  free(program_buffer);

  //Build program
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (error < 0) {
    clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    program_log = (char*) malloc(log_size + 1);
    program_log[log_size] = '\0';
    clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
    printf("%s\n", program_log);
    free(program_log);
    exit(EXIT_FAILURE);
  }

  return program;
}
