#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../src/cl_host_tools.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

int main(){
  cl_host_tools_get_device_info();
  return 0;
}
