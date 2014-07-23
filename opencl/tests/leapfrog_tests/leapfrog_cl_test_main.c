#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "leapfrog_params.h"
#include "../src/cl_host_tools.h"
#include "leapfrog_cl_test.h"
#include "leapfrog_c_test.h"

int main(int argc, char *argv[])
{
  float dt = .001;

  float xi[N];
  float yi[N];
  float zi[N];  
  float vxi[N];
  float vyi[N];
  float vzi[N];

  float x[N];
  float y[N];
  float z[N];  
  float vx[N];
  float vy[N];
  float vz[N];

  float x_dev[N];
  float y_dev[N];
  float z_dev[N];  
  float vx_dev[N];
  float vy_dev[N];
  float vz_dev[N];
  time_t cl_start, cl_end, c_start, c_end;
  bool pass_test;
  int i;
  
  srand(time(NULL));
  for (i = 0; i < N; i++) {
    xi[i]= cl_host_tools_random_float();
    yi[i]= cl_host_tools_random_float();
    zi[i]= cl_host_tools_random_float();
    vxi[i]= cl_host_tools_random_float();
    vyi[i]= cl_host_tools_random_float();
    vzi[i]= cl_host_tools_random_float();

    x[i] = xi[i];
    y[i] = yi[i];
    z[i] = zi[i];
    vx[i] = vxi[i];
    vy[i] = vyi[i];
    vz[i] = vzi[i];


    x_dev[i] = x[i];
    y_dev[i] = y[i];
    z_dev[i] = z[i];
    vx_dev[i] = vx[i];
    vy_dev[i] = vy[i];
    vz_dev[i] = vz[i];
   }

  cl_start = clock();
  leapfrog_cl_test(x_dev,y_dev,z_dev,vx_dev,vy_dev,vz_dev,dt);
  cl_end = clock();

  c_start = clock();
  leapfrog_c_test(x,y,z,vx,vy,vz,dt);
  c_end = clock();

  pass_test = leapfrog_c_test_check(x,y,z,vx,vy,vz,x_dev,y_dev,z_dev,vx_dev,vy_dev,vz_dev);

  printf(" Number of timesteps = %d\n", RUNS);
  printf(" GPU time =  %f\n", (double)(cl_end - cl_start) / CLOCKS_PER_SEC);
  printf(" CPU time =  %f\n", (double)(c_end - c_start) / CLOCKS_PER_SEC);
  printf(" Test passed flag = %u\n", (unsigned int)pass_test);

  int num_to_print = N;
  printf("i +++  x_initial +++ x_device +++ x_host\n");
  for (int i = 0; i < num_to_print; i++)
    printf("%d +++ %.7f +++ %.7f +++ %.7f\n",i,xi[i],x_dev[i],x[i]);

  return 0;
}
