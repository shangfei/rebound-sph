#ifndef CL_HOST_TOOLS_H
#define CL_HOST_TOOLS_H

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_CL_FILES 100

float cl_host_tools_random_float();
cl_device_id cl_host_tools_create_device();
cl_program cl_host_tools_create_program(cl_context, cl_device_id, const char**, const char*, int);
cl_uint cl_host_tools_get_num_compute_units(cl_device_id);
size_t cl_host_tools_get_max_work_group_size(cl_device_id);
cl_ulong cl_host_tools_get_local_mem_size(cl_device_id);
cl_ulong cl_host_tools_get_global_mem_size(cl_device_id);
size_t cl_host_tools_get_max_work_item_size(cl_device_id, int);
void cl_host_tools_get_device_info();

#endif
