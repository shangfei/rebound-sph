#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cl_host_tools.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

cl_float cl_host_tools_normaldistribution2_rsq;		/**< Used for speedup**/ 
cl_float cl_host_tools_normaldistribution2_v2;		/**< Used for speedup**/
cl_int 	cl_host_tools_normaldistribution2_ready = 0;	/**< Used for speedup**/

/* cache sei integrator coefficients */
void cl_host_tools_integrator_cache_coefficients(
				        float* OMEGA,
					float* OMEGAZ,
					float* sindt,
					float* tandt,
					float* sindtz,
					float* tandtz,
					float* dt
				        ){
  if (*OMEGAZ == -1)
    *OMEGAZ=*OMEGA;

  *sindt = (float) sin((double)( *OMEGA*(-*dt/2.f) ));
  *tandt = (float) tan((double)( *OMEGA*(-*dt/4.f) ));
  *sindtz = (float) sin((double)( *OMEGAZ*(-*dt/2.f) ));
  *tandtz = (float) tan((double)( *OMEGAZ*(-*dt/4.f) ));
}

const char * cl_host_tools_get_error_string(cl_int error){
  switch(error){
   
    // run-time and JIT compiler errors
  case 0: return "CL_SUCCESS";
  case -1: return "CL_DEVICE_NOT_FOUND";
  case -2: return "CL_DEVICE_NOT_AVAILABLE";
  case -3: return "CL_COMPILER_NOT_AVAILABLE";
  case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5: return "CL_OUT_OF_RESOURCES";
  case -6: return "CL_OUT_OF_HOST_MEMORY";
  case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8: return "CL_MEM_COPY_OVERLAP";
  case -9: return "CL_IMAGE_FORMAT_MISMATCH";
  case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -11: return "CL_BUILD_PROGRAM_FAILURE";
  case -12: return "CL_MAP_FAILURE";
  case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case -15: return "CL_COMPILE_PROGRAM_FAILURE";
  case -16: return "CL_LINKER_NOT_AVAILABLE";
  case -17: return "CL_LINK_PROGRAM_FAILURE";
  case -18: return "CL_DEVICE_PARTITION_FAILED";
  case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
    
    // compile-time errors
  case -30: return "CL_INVALID_VALUE";
  case -31: return "CL_INVALID_DEVICE_TYPE";
  case -32: return "CL_INVALID_PLATFORM";
  case -33: return "CL_INVALID_DEVICE";
  case -34: return "CL_INVALID_CONTEXT";
  case -35: return "CL_INVALID_QUEUE_PROPERTIES";
  case -36: return "CL_INVALID_COMMAND_QUEUE";
  case -37: return "CL_INVALID_HOST_PTR";
  case -38: return "CL_INVALID_MEM_OBJECT";
  case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40: return "CL_INVALID_IMAGE_SIZE";
  case -41: return "CL_INVALID_SAMPLER";
  case -42: return "CL_INVALID_BINARY";
  case -43: return "CL_INVALID_BUILD_OPTIONS";
  case -44: return "CL_INVALID_PROGRAM";
  case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46: return "CL_INVALID_KERNEL_NAME";
  case -47: return "CL_INVALID_KERNEL_DEFINITION";
  case -48: return "CL_INVALID_KERNEL";
  case -49: return "CL_INVALID_ARG_INDEX";
  case -50: return "CL_INVALID_ARG_VALUE";
  case -51: return "CL_INVALID_ARG_SIZE";
  case -52: return "CL_INVALID_KERNEL_ARGS";
  case -53: return "CL_INVALID_WORK_DIMENSION";
  case -54: return "CL_INVALID_WORK_GROUP_SIZE";
  case -55: return "CL_INVALID_WORK_ITEM_SIZE";
  case -56: return "CL_INVALID_GLOBAL_OFFSET";
  case -57: return "CL_INVALID_EVENT_WAIT_LIST";
  case -58: return "CL_INVALID_EVENT";
  case -59: return "CL_INVALID_OPERATION";
  case -60: return "CL_INVALID_GL_OBJECT";
  case -61: return "CL_INVALID_BUFFER_SIZE";
  case -62: return "CL_INVALID_MIP_LEVEL";
  case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  case -64: return "CL_INVALID_PROPERTY";
  case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
  case -66: return "CL_INVALID_COMPILER_OPTIONS";
  case -67: return "CL_INVALID_LINKER_OPTIONS";
  case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
  case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
  case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
  case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
  case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
  case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
  default: return "Unknown OpenCL error";
  }
}

cl_float cl_host_tools_uniform(cl_float min, cl_float max){
  return cl_host_tools_random_float()*(max-min)+min;
}

cl_float cl_host_tools_random_float(){
  return (cl_float)rand() / (cl_float)RAND_MAX;
}

cl_float cl_host_tools_powerlaw(cl_float min, cl_float max, cl_float slope){
  float y = cl_host_tools_uniform(0.f,1.f);
  return  pow( (pow(max,slope+1.)-pow(min,slope+1.))*y+pow(min,slope+1.), 1./(slope+1.));
}


cl_float cl_host_tools_normal(cl_float variance){
  if (cl_host_tools_normaldistribution2_ready==1){
    cl_host_tools_normaldistribution2_ready = 0;
    return cl_host_tools_normaldistribution2_v2*sqrt(-2.*log(cl_host_tools_normaldistribution2_rsq)/cl_host_tools_normaldistribution2_rsq*variance);
  }

  cl_float v1,v2,rsq=1.f;
  while (rsq>=1. || rsq<1.0e-12){
    v1=2.f*cl_host_tools_uniform(0.f,1.f)-1.f;
    v2=2.f*cl_host_tools_uniform(0.f,1.f)-1.f;
    rsq=v1*v1+v2*v2;
  }
  cl_host_tools_normaldistribution2_ready = 1;
  cl_host_tools_normaldistribution2_rsq = rsq;
  cl_host_tools_normaldistribution2_v2 = v2;
  return v1*sqrt(-2.f*log(rsq)/rsq*variance);
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

cl_program cl_host_tools_create_program(     
					cl_context context,
					cl_device_id device,
					const char** file_names, 
					const char* options,  
					int num_files
				     ){

  cl_program program;
  FILE *program_handle;
  char *program_log;
  char *program_buffer[MAX_CL_FILES];
  size_t program_size[MAX_CL_FILES];
  size_t log_size;
  cl_int error;

  if (num_files > MAX_CL_FILES){
    fprintf(stderr, "The number of .CL files used exceeds MAX_CL_FILES. Please increase MAX_CL_FILES.");
    exit(EXIT_FAILURE);
  }

  //read in program
  for(int i=0; i< num_files; i++) {
    program_handle = fopen(file_names[i], "r");
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
  program = clCreateProgramWithSource(context, num_files, (const char**)program_buffer, program_size, &error);
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
