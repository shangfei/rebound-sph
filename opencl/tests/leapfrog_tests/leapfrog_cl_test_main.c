#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 32768

#include "leapfrog_cl_test.h"
#include "leapfrog_c_test.h"

int main(int argc, char *argv[])
{

  float dt = .001;
  float x[N];
  float y[N];
  float z[N];  
  float vx[N];
  float vy[N};
  float vz[N];
  
  srand(time(NULL));
  for (i = 0; i < N; i++) {
    x[i]= cl_host_tools_random_float();
    y[i]= cl_host_tools_random_float();
    z[i]= cl_host_tools_random_float();
    vx[i]= cl_host_tools_random_float();
    vy[i]= cl_host_tools_random_float();
    vz[i]= cl_host_tools_random_float();
  }

  time_t cl_start, cl_end, c_start, c_end;

  cl_start = clock();
  leapfrog_cl_test();
  cl_end = clock();

  c_start = clock();
  leapfrog_c_test();
  c_end = clock();

  printf(" GPU time =  %1u\n", (cl_end - cl_start));
  printf(" CPU time =  %1u\n", (cl_end - cl_start));

  return 0;
}
