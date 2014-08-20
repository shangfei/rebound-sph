#ifndef CL_INIT_H
#define CL_INIT_H

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

void cl_init_create_device();
void cl_init_create_context();
void cl_init_create_program();
void cl_init_create_command_queue();
void cl_init_print_thread_info();
void cl_init_create_kernels();
void cl_init_create_buffers();
void cl_init_free_globals();
void cl_init_set_kernel_arg_tree_kernel();
void cl_init_set_kernel_arg_tree_gravity_kernel();
void cl_init_set_kernel_arg_tree_sort_kernel();
void cl_init_set_kernel_arg_force_gravity_kernel();
void cl_init_set_kernel_arg_collisions_search_kernel();
void cl_init_set_kernel_arg_collisions_resolve_kernel();
void cl_init_set_kernel_arg_tree_collisions_kernel();
void cl_init_set_kernel_arg_tree_kernel_no_mass();
void cl_init_set_kernel_arg_boundaries_kernel();
void cl_init_set_kernel_arg_integrator_part1_kernel();
void cl_init_set_kernel_arg_integrator_part2_kernel();
void cl_init_enqueue_integrator_part1_kernel();
void cl_init_enqueue_boundaries_kernel();
void cl_init_enqueue_tree_kernel();
void cl_init_enqueue_tree_gravity_kernel();
void cl_init_enqueue_tree_sort_kernel();
void cl_init_enqueue_force_gravity_kernel();
void cl_init_enqueue_integrator_part2_kernel();
void cl_init_enqueue_tree_kernel_no_mass();
void cl_init_enqueue_tree_collisions_kernel();
void cl_init_enqueue_collisions_search_kernel();
void cl_init_enqueue_collisions_resolve_kernel();


#endif
