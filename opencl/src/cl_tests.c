#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "cl_init.h"
#include "cl_tests.h"
#include "cl_host_tools.h"
#include "cl_globals.h"

int cl_tests_particles_intersect(float x1, float y1, float z1, float r1, float x2, float y2, float z2, float r2){

  float dx = (x1-x2);
  float dy = (y1-y2);
  float dz = (z1-z2);
  float rp = r1 + r2;

  if ( dx*dx + dy*dy + dz*dz <= rp*rp )
    return 1;
  else
    return 0;

}


void cl_tests_print_particle_info(
				  int step,
				  cl_char * message,
				  int transfer_from_device
				  ){
  
  if(transfer_from_device == 1){
    cl_tests_read_particle_positions_from_device();
    cl_tests_read_particle_velocities_from_device();
    cl_tests_read_particle_accelerations_from_device();
    cl_tests_read_particle_masses_from_device();
  }

  printf( "\n PARTICLE INFO FOR: %s \n", message);
  printf( "ITERATION: %d \n", step);
  for (int i = 0; i < num_nodes_host + 1; i++){
    if (i < num_bodies_host){
      printf(" %d %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f\n", i, x_host[i], y_host[i], z_host[i], vx_host[i], vy_host[i], vz_host[i], ax_host[i], ay_host[i], az_host[i],  mass_host[i], rad_host[i]);
    }
    else
      printf(" %d %.15f %.15f %.15f %.15f %.15f %.15f %.15f\n", i, x_host[i], y_host[i], z_host[i], 0., 0., 0.,  mass_host[i]);      
  }
}


void cl_tests_print_tree_info(
			      int step,
			      cl_char * message,
			      int transfer_from_device
			      ){

  if(transfer_from_device == 1){
    cl_tests_read_tree_vars_from_device();
  }
  printf( "\n TREE INFO FOR: %s \n", message);
  printf( "ITERATION: %d \n", step);
  
  printf("\n++++++COUNT_HOST++++++\n");
  for (int i = bottom_node_host; i < num_nodes_host + 1; i++){
    printf(" %d %d \n", i, count_host[i]);
  }

  printf("\n++++++SORT_HOST++++++\n");
  for (int i = 0; i < num_nodes_host + 1; i++){
    printf(" %d %d \n", i, sort_host[i]);
  }

  printf("\n++++++START_HOST++++++\n");
  for (int i = bottom_node_host; i < num_nodes_host + 1; i++){
    printf(" %d %d \n", i, start_host[i]);
  }

  printf("\n++++++COLLISIONS_HOST+++++++\n");
  for (int i = 0; i < num_bodies_host * (nghostx + 1) * (nghosty + 1) * (nghostz + 1); i++){
    printf(" %d %d \n", i, collisions_host[i]);
  }
  
  printf("\n++++++TREE++++++\n");

  for (int node = bottom_node_host; node < num_nodes_host + 1; node++){
    printf("+++++NODE %d+++++:",node);
    for (int child = 0; child < 8; child++)
      printf(" %d ", children_host[node*8 + child]);
    printf("\n");
  }

}


void cl_tests_read_tree_vars_from_device(){
  error = clEnqueueReadBuffer(queue, children_buffer, CL_TRUE, 0, sizeof(cl_int) * 8 * (num_nodes_host + 1), children_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (children_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, collisions_buffer, CL_TRUE, 0, sizeof(cl_int) * num_bodies_host * (nghostx + 1) * (nghosty + 1) * (nghostz + 1), collisions_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (collisions_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, bottom_node_buffer, CL_TRUE, 0, sizeof(cl_int), &bottom_node_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (bottom_node_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, maxdepth_buffer, CL_TRUE, 0, sizeof(cl_int), &maxdepth_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (maxdepth_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, start_buffer, CL_TRUE, 0, sizeof(cl_int) * (num_nodes_host + 1), start_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (start_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, sort_buffer, CL_TRUE, 0, sizeof(cl_int) * (num_nodes_host + 1), sort_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (sort_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, count_buffer, CL_TRUE, 0, sizeof(cl_int) * (num_nodes_host + 1), count_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (count_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

}

void cl_tests_read_particle_masses_from_device(){

 error = clEnqueueReadBuffer(queue, mass_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_nodes_host + 1), mass_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (mass_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

}

void cl_tests_read_particle_accelerations_from_device(){

  error = clEnqueueReadBuffer(queue, ax_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_bodies_host), ax_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr, "clEqueueReadBuffer ERROR (ax_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, ay_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_bodies_host), ay_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr, "clEqueueReadBuffer ERROR (ay_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, az_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_bodies_host), az_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr, "clEqueueReadBuffer ERROR (az_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

}

void cl_tests_read_particle_velocities_from_device(){

  error = clEnqueueReadBuffer(queue, vx_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_bodies_host), vx_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr, "clEqueueReadBuffer ERROR (vx_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, vy_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_bodies_host), vy_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr, "clEqueueReadBuffer ERROR (vy_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, vz_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_bodies_host), vz_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr, "clEqueueReadBuffer ERROR (vz_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

}

void cl_tests_read_particle_positions_from_device(){

 // Read the results

  error = clEnqueueReadBuffer(queue, x_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_nodes_host + 1), x_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (x_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, y_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_nodes_host + 1), y_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (y_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  error = clEnqueueReadBuffer(queue, z_buffer, CL_TRUE, 0, sizeof(cl_float) * (num_nodes_host + 1), z_host, 0, NULL, NULL);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clEnqueueReadBuffer ERROR (z_buffer): %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

}

void cl_tests_boundaries_get_ghostbox(
				         cl_int i,
					 cl_int j,
					 cl_int k,
					 cl_float *shiftx,
					 cl_float *shifty,
					 cl_float *shiftz,
					 cl_float *shiftvx,
					 cl_float *shiftvy,
					 cl_float *shiftvz
				        ){

  *shiftvx = 0.f;
  *shiftvy = -1.5f*(float)i*(OMEGA_host)*(boxsize_host);
  *shiftvz = 0.f;
  float shift = (i == 0) ? 0.f : -fmod( (*shiftvy)*(t_host) - ((i>0) - (i<0)) * (boxsize_host)/2.f, (boxsize_host)) - ((i>0) - (i<0))*(boxsize_host)/2.f;
  *shiftx = (boxsize_host)*(float)i;
  *shifty = (boxsize_host)*(float)j-shift;
  *shiftz = (boxsize_host)*(float)k;

}

void cl_tests_force_gravity_test(){

  cl_double *ax_direct;
  cl_double *ay_direct;
  cl_double *az_direct;
  
  ax_direct = (cl_double *) malloc( sizeof(cl_double) * num_bodies_host );
  ay_direct = (cl_double *) malloc( sizeof(cl_double) * num_bodies_host );
  az_direct = (cl_double *) malloc( sizeof(cl_double) * num_bodies_host );

  cl_tests_cpu_force_gravity_direct_summation(ax_direct, ay_direct, az_direct);
 
  cl_init_enqueue_tree_kernel();
  cl_init_enqueue_tree_gravity_kernel();
  cl_init_enqueue_tree_sort_kernel();
  cl_init_enqueue_force_gravity_kernel();
  error = clFinish(queue);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clFinish ERROR: %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }

  cl_tests_read_particle_accelerations_from_device();
  for (int i = 0; i < num_bodies_host; i++){
    if (i < num_bodies_host){
      printf(" %d %.15f %.15f %.15f %.15f %.15f %.15f \n", i, ax_host[i], ay_host[i], az_host[i], ax_direct[i], ay_direct[i], az_direct[i]);
    }
  }
  
  free(ax_direct);
  free(ay_direct);
  free(az_direct);
}


void cl_tests_collisions_search_test(){

  cl_int * collisions_cpu_host;
  collisions_cpu_host =  (cl_int *) malloc( sizeof(cl_int) * num_bodies_host * (nghostx + 1) * (nghosty + 1) * (nghostz + 1) );

  cl_tests_cpu_collisions_search_direct(collisions_cpu_host);
  cl_init_enqueue_tree_kernel_no_mass();
  cl_init_enqueue_tree_collisions_kernel();
  cl_init_enqueue_tree_sort_kernel();
  cl_init_enqueue_collisions_search_kernel();
   error = clFinish(queue);
  if(error != CL_SUCCESS) {
    fprintf(stderr,"clFinish ERROR: %s\n",cl_host_tools_get_error_string(error));
    exit(EXIT_FAILURE);
  }
  cl_tests_read_tree_vars_from_device();

  printf("\n++++++COLLISIONS_HOST-------COLLISIONS_CPU_HOST+++++++\n");
  for (int i = 0; i < num_bodies_host * (nghostx + 1) * (nghosty + 1) * (nghostz + 1); i++){
    printf(" %d %d %d \n", i, collisions_host[i], collisions_cpu_host[i]);
  }
  
  free(collisions_cpu_host);

}


void cl_tests_cpu_force_gravity_direct_summation(
						      cl_double * ax_direct,
						      cl_double * ay_direct,
						      cl_double * az_direct
						     ){

  float shiftx,shifty,shiftz,shiftvx,shiftvy,shiftvz;

  for (int i = 0; i < num_bodies_host; i++){
    ax_direct[i] = 0;
    ay_direct[i] = 0;
    az_direct[i] = 0;
  }


    for (int gbx = -1; gbx <= 1; gbx++){
      for (int gby = -1; gby <= 1; gby++){
	for (int gbz = -1; gbz <= 1; gbz++){
	  cl_tests_boundaries_get_ghostbox(gbx,gby,gbz,&shiftx,&shifty,&shiftz,&shiftvx,&shiftvy,&shiftvz);
	  for (int i = 0; i < num_bodies_host; i++){
	    for (int j = 0; j < num_bodies_host; j++){
	      if (i == j) continue;
	      cl_double dx = x_host[i] + shiftx - x_host[j];
	      cl_double dy = y_host[i] + shifty - y_host[j];
	      cl_double dz = z_host[i] + shiftz - z_host[j];
	      cl_double r = sqrt(dx*dx + dy*dy + dz*dz + softening2_host);
	      cl_double prefact = -G_host/(r*r*r)*mass_host[j];
	      ax_direct[i] += prefact*dx;
	      ay_direct[i] += prefact*dy;
	      az_direct[i] += prefact*dz;
	    }
	  }  
	}
      }
    }
}

void cl_tests_cpu_collisions_search_direct(
					        cl_int * collisions_cpu_host
					      )

{
  float shiftx,shifty,shiftz,shiftvx,shiftvy,shiftvz;
  float body1_x, body1_y, body1_z, body1_vx, body1_vy, body1_vz;
  float body2_x, body2_y, body2_z, body2_vx, body2_vy, body2_vz;
  float sr, r2, dx, dy, dz, dvx, dvy, dvz;

  int gbx_offset = num_bodies_host;
  int gby_offset = gbx_offset*3;
  int gbz_offset = gby_offset*3;

  for (int gbx=-1; gbx<=1; gbx++){
    for (int gby=-1; gby<=1; gby++){
      for (int gbz=-1; gbz<=1; gbz++){
	// Loop over all particles

	cl_tests_boundaries_get_ghostbox(gbx,gby,gbz,&shiftx,&shifty,&shiftz,&shiftvx,&shiftvy,&shiftvz);
	for (int i=0;i<num_bodies_host;i++){
	  collisions_cpu_host[i + gbx_offset*(gbx+1) + gby_offset*(gby+1) + gbz_offset*(gbz+1)] = -1;
	  body1_x = x_host[i] + shiftx;
	  body1_y = y_host[i] + shifty;
	  body1_z = z_host[i] + shiftz;
	  body1_vx = vx_host[i] + shiftvx;
	  body1_vy = vy_host[i] + shiftvy;
	  body1_vz = vz_host[i] + shiftvz;
	  for (int j=0;j<num_bodies_host;j++){
	    // Do not collide particle with itself.

	    if (i==j) continue;
	    body2_x = x_host[j];
	    body2_y = y_host[j];
	    body2_z = z_host[j];
	    body2_vx = vx_host[j];
	    body2_vy = vy_host[j];
	    body2_vz = vz_host[j];
	    dx = body1_x - body2_x;
	    dy = body1_y - body2_y;
	    dz = body1_z - body2_z;
	    sr = rad_host[i] + rad_host[j];
	    r2 = dx*dx + dy*dy + dz*dz;

	    // Check if particles are overlapping
	    if (r2>sr*sr)
	      continue;
	    dvx = body1_vx - body2_vx;
	    dvy = body1_vy - body2_vy;
	    dvz = body1_vz - body2_vz;
	    // Check if particles are approaching each other
	    if (dvx*dx + dvy*dy + dvz*dz >0)
	      continue;
	    collisions_cpu_host[i + gbx_offset*(gbx+1) + gby_offset*(gby+1) + gbz_offset*(gbz+1)] = j;
	  }
	}
      }
    }
 }
}
					
