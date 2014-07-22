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

  float x_dev[N];
  float y_dev[N];
  float z_dev[N];  
  float vx_dev[N];
  float vy_dev[N};
  float vz_dev[N];
  time_t cl_start, cl_end, c_start, c_end;
  bool pass_test;

  srand(time(NULL));
  
  for (i = 0; i < N; i++) {
    x[i]= cl_host_tools_random_float();
    y[i]= cl_host_tools_random_float();
    z[i]= cl_host_tools_random_float();
    vx[i]= cl_host_tools_random_float();
    vy[i]= cl_host_tools_random_float();
    vz[i]= cl_host_tools_random_float();
    x_dev[i] = x[i];
    y_dev[i] = y[i];
    z_dev[i] = z[i];
    vx_dev[i] = vx[i];
    vy_dev[i] = vy[i];
    vz_dev[i] = vz[i];
  }

  cl_start = clock();
  leapfrog_cl_test(dt,x_dev,y_dev,z_dev,vx_dev,vy_dev,vz_dev);
  cl_end = clock();

  c_start = clock();
  leapfrog_c_test(dt,x,y,z,vx,vy,vz);
  c_end = clock();

  pass_test = leapfrog_c_test_check(x,y,z,vx,vy,vz,x_dev,y_dev,z_dev,vx_dev,vy_dev,vz_dev);

  printf(" GPU time =  %1u\n", (cl_end - cl_start));
  printf(" CPU time =  %1u\n", (cl_end - cl_start));
  printf(" Test passed flag = %u\n", pass_test);

  return 0;
}
