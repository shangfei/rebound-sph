#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "cl_shearing_sheet.h"

int main(int argc, char *argv[])
{
  int num_bodies;
  int num_threads_tree_kernel;
  int num_threads_tree_gravity_kernel;
  int num_threads_tree_sort_kernel;
  int num_threads_force_gravity_kernel;
  int num_threads_collisions_search_kernel;
  int num_threads_collisions_resolve_kernel;
  int num_threads_boundaries_kernel;
  int num_threads_integrator_kernel;

  if (argc != 10){
    num_bodies = 32768;
    num_threads_tree_gravity_kernel = 128;
    num_threads_tree_kernel = 128;
    num_threads_tree_sort_kernel = 128;
    num_threads_force_gravity_kernel = 128;
    num_threads_collisions_search_kernel = 128;
    num_threads_collisions_resolve_kernel = 128;
    num_threads_boundaries_kernel = 128;
    num_threads_integrator_kernel = 128;
  }
  else{
    num_bodies = atoi(argv[1]);
    num_threads_tree_kernel = atoi(argv[2]);
    num_threads_tree_gravity_kernel = atoi(argv[3]);
    num_threads_tree_sort_kernel = atoi(argv[4]);
    num_threads_force_gravity_kernel = atoi(argv[5]);
    num_threads_collisions_search_kernel = atoi(argv[6]);
    num_threads_collisions_resolve_kernel = atoi(argv[7]);
    num_threads_boundaries_kernel = atoi(argv[8]);
    num_threads_integrator_kernel = atoi(argv[9]);
  }

  cl_problem_init(num_bodies, num_threads_tree_kernel, num_threads_tree_gravity_kernel,num_threads_tree_sort_kernel, num_threads_force_gravity_kernel, num_threads_collisions_search_kernel, num_threads_collisions_resolve_kernel, num_threads_boundaries_kernel, num_threads_integrator_kernel);
  return 0;
}
