#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cl_host_tools.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

float cl_host_tools_random_float(){
  return (float)rand() / (float)RAND_MAX;
}

/* Returns the number of multiprocessors on the device */
cl_uint cl_host_tools_get_num_compute_units(cl_device_id device){
  cl_uint mps;
  cl_int err;
  err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &mps, NULL);
  if(err < 0){
    fprintf(stderr,"Could not fetch CL_DEVICE_MAX_COMPUTE_UNITS");
    exit(1);
  }
  return mps;
}

size_t cl_host_tools_get_max_work_group_size(cl_device_id device){
  size_t group_size;
  cl_int err;
  err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(group_size), &group_size, NULL);
  if(err < 0){
    fprintf(stderr,"Could not fetch CL_DEVICE_MAX_WORK_GROUP_SIZE");
    exit(1);
  }  
  return group_size;
}

cl_ulong cl_host_tools_get_local_mem_size(cl_device_id device){
  cl_ulong mem_size;
  cl_int err;
  err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
  if(err < 0){
    fprintf(stderr,"Could not fetch CL_DEVICE_LOCAL_MEM_SIZE");
    exit(1);
  } 
  return mem_size;
}

cl_ulong cl_host_tools_get_global_mem_size(cl_device_id device){
  cl_ulong mem_size;
  cl_int err;
  err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
  if(err < 0){
    fprintf(stderr,"Could not fetch CL_DEVICE_GLOBAL_MEM_SIZE");
    exit(1);
  } 
  return mem_size;
}

size_t cl_host_tools_get_max_work_item_size(cl_device_id device, int dim){
  if(dim < 0 || dim > 2){
    fprintf(stderr,"Invalid dimension (dim)");
    exit(1);
  }
  size_t workitem_size[3];
  cl_int err;
  err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
  if(err < 0){
    fprintf(stderr,"Could not fetch CL_DEVICE_MAX_WORK_ITEM_SIZES");
    exit(1);
  }     
  return workitem_size[dim];
}


void cl_host_tools_get_device_info(){
  int i;
  cl_platform_id platforms[100];
  cl_uint platforms_n = 0;
  int plat;

  clGetPlatformIDs(100, platforms, &platforms_n);
  printf("====================================\n");
  printf("=== %d OpenCL platform(s) found: ===\n", platforms_n);
  printf("====================================\n\n");

  for (plat=0; plat<platforms_n; ++plat)
    {
      cl_device_id devices[100];
      cl_uint devices_n = 0;
      cl_char buffer[10240];

      printf("-----------------------------------------------\nPLATFORM: %d\n", plat);
      clGetPlatformInfo(platforms[plat], CL_PLATFORM_PROFILE, 10240, buffer, NULL);
      printf("PROFILE = %s\n", buffer);
      clGetPlatformInfo(platforms[plat], CL_PLATFORM_VERSION, 10240, buffer, NULL);
      printf("VERSION = %s\n", buffer);
      clGetPlatformInfo(platforms[plat], CL_PLATFORM_NAME, 10240, buffer, NULL);
      printf("NAME = %s\n", buffer);
      clGetPlatformInfo(platforms[plat], CL_PLATFORM_VENDOR, 10240, buffer, NULL);
      printf("VENDOR = %s\n", buffer);
      clGetPlatformInfo(platforms[plat], CL_PLATFORM_EXTENSIONS, 10240, buffer, NULL);
      printf("EXTENSIONS = %s\n", buffer);
    
      cl_uint dtype = CL_DEVICE_TYPE_ALL;
      clGetDeviceIDs(platforms[plat], dtype, 100, devices, &devices_n);
      
      for (i=0; i<devices_n; i++)
	{
	  cl_char buffer[10240];
	  cl_uint buf_uint;
	  cl_ulong buf_ulong;
	  printf("***********************************************\nPLATFORM: %d DEVICE: %d\n", plat, i);
	  clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
	  printf("DEVICE_NAME = %s\n", buffer);
	  clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
	  printf("DEVICE_VENDOR = %s\n", buffer);
	  clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
	  printf("DEVICE_VERSION = %s\n", buffer);
	  clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL);
	  printf("DRIVER_VERSION = %s\n", buffer);
	  clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL);
	  printf("DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
	  clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL);
	  printf("DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
	  clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
	  printf("DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
	  clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
	  printf("DEVICE_MAX_WORK_GROUP_SIZE = %llu\n", (unsigned long long)buf_ulong);
	  clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(buf_uint), &buf_uint, NULL);
	  printf("CL_DEVICE_MAX_CONSTANT_ARGS = %u\n",(unsigned int)buf_uint);
	  clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
	  printf("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE = %llu\n", (unsigned long long)buf_ulong);
	  clGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
	  printf("CL_DEVICE_LOCAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
	  size_t workitem_dims;
	  clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(workitem_dims), &workitem_dims, NULL);
	  printf("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:\t%u\n", (unsigned int) workitem_dims); 
	  size_t workitem_size[3];
	  clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
	  printf("CL_DEVICE_MAX_WORK_ITEM_SIZES:\t%u / %u / %u \n", (unsigned int)workitem_size[0], (unsigned int)workitem_size[1], (unsigned int)workitem_size[2]);
	}
      printf("-----------------------------------------------\n\n");
    }
}

cl_device_id cl_host_tools_create_device(){

  cl_platform_id platform;
  cl_device_id device;
  int error;
 
  //identify the first platform
  error = clGetPlatformIDs(1, &platform, NULL);
  if (error < 0){
    fprintf(stderr,"Couldn't find a platform");
    exit(EXIT_FAILURE);
  }

  //get first GPU device available
  error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  if(error == CL_DEVICE_NOT_FOUND){
    printf("Could not find a GPU device. Using CPU instead...");
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
  }
  if(error < 0) {
    fprintf(stderr,"Couldn't access any device");
    exit(EXIT_FAILURE);
  }
  return device;

}

cl_program cl_host_tools_create_program(cl_context context, 
					cl_device_id device, 
					const char* filename [], 
					const char options [],  
					int num_files){

  cl_program program;
  FILE *program_handle;
  char *program_log;
  char *program_buffer[MAX_CL_FILES];
  size_t program_size;
  size_t log_size;
  cl_int error;

  if (num_files > MAX_CL_FILES){
    fprintf(stderr, "The number of .CL files used exceeds MAX_CL_FILES. Please increase MAX_CL_FILES.");
    exit(EXIT_FAILURE);
  }

  //read in program
  for(int i=0; i< num_files; i++) {
    program_handle = fopen(file_name[i], "r");
    if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);   
    }
    fseek(program_handle, 0, SEEK_END);
    program_size[i] = ftell(program_handle);
    rewind(program_handle);
    program_buffer[i] = (char*)malloc(program_size[i]+1);
    program_buffer[i][program_size[i]] = '\0';
    fread(program_buffer[i], sizeof(char), program_size[i], 
	  program_handle);
    fclose(program_handle);
  }
  
  //Create program from file
  program = clCreateProgramWithSource(context, num_files, (const char**)program_buffer, program_size, &err);
  if(error < 0){
    fprintf(stderr,"Couldn't create the program");
    exit(EXIT_FAILURE);
  }
  
  error = clBuildProgram(program, 1, &device, options, NULL, NULL);
  if (error < 0) {
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    program_log = (char*) malloc(log_size + 1);
    program_log[log_size] = '\0';
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
    printf("%s\n", program_log);
    free(program_log);
    exit(EXIT_FAILURE);
  }

  for(int i = 0; i < num_files; i++)
      free(program_buffer[i]);

  return program;
}
