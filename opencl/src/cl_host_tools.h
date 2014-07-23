#ifndef CL_HOST_TOOLS_H
#define CL_HOST_TOOLS_H

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

float cl_host_tools_random_float();
cl_device_id cl_host_tools_create_device();
cl_program cl_host_tools_create_program(cl_context, cl_device_id, const char*);

#endif
