#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "tree_cl_test.h"

int main(int argc, char *argv[])
{
  int num_bodies;
  int num_threads;
  if (argc != 3){
    num_bodies = 32768;
    num_threads = 128;
  }
  else{
    num_bodies = atoi(argv[1]);
    num_threads = atoi(argv[2]);
  }
  tree_cl_test(num_bodies, num_threads);
  return 0;
}
