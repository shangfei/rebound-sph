#ifndef LEAPFROG_C_TEST_H
#define LEAPFROG_C_TEST_H

#include <stdbool.h> 

void cl_integrator_leapfrog_part1(float [], float [], float [], float [], float [], float [], float);
void cl_integrator_leapfrog_part2(float [], float [], float [],float [],float [], float [], float);
bool leapfrog_c_test_compare_floats(float, float, float);
bool leapfrog_c_test_check(float [], float [], float [], float [], float [], float [], float [], float [], float [], float [], float [], float []);
void leapfrog_c_test(float [], float [], float [], float [], float [], float [], float);

#endif
