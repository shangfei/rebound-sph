#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../../src/cl_host_tools.h"
#include "tree_cl_test.h"

int main(int argc, char *argv[])
{

  float *x;
  float *y;
  float *z;
  float boxsize = 10;
  float rootx = 0.f;
  float rooty = 0.f;
  float rootz = 0.f;
  int num_bodies = 32768;
  
  x = (float *)malloc((num_bodies)*sizeof(float));
  y = (float *)malloc((num_bodies)*sizeof(float));
  z = (float *)malloc((num_bodies)*sizeof(float));
  
  srand(time(NULL));
  for (int i = 1; i < num_bodies; i++){
    x[i] = rootx - boxsize/2.f + boxsize*cl_host_tools_random_float();
    y[i] = rooty - boxsize/2.f + boxsize*cl_host_tools_random_float();
    z[i] = rootz - boxsize/2.f + boxsize*cl_host_tools_random_float();
  }

  tree_cl_test(x,y,z,&boxsize,&rootx,&rooty,&rootz,&num_bodies);

  free(x);
  free(y);
  free(z);

  return 0;
}
