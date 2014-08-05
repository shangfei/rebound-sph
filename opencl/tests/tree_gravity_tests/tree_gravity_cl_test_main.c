#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "tree_gravity_cl_test.h"

int main(int argc, char *argv[])
{
  int num_bodies;
  int num_threads_tree_kernel;
  int num_threads_tree_gravity_kernel;

  if (argc != 4){
    num_bodies = 32768;
    num_threads_tree_gravity_kernel = 128;
    num_threads_tree_kernel = 128;
  }

  else{
    num_bodies = atoi(argv[1]);
    num_threads_tree_kernel = atoi(argv[2]);
    num_threads_tree_gravity_kernel = atoi(argv[3]);
  }

  tree_gravity_cl_test(num_bodies, num_threads_tree_kernel, num_threads_tree_gravity_kernel);
  return 0;
}
