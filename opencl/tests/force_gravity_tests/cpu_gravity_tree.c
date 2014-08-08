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

#include "cl_gpu_defns.h"

void cpu_gravity_calculate_acceleration_for_particle(	
						           cl_float * x_dev, 
							   cl_float * y_dev,
							   cl_float * z_dev,
							   cl_float * ax_dev,
							   cl_float * ay_dev,
							   cl_float * az_dev,
							   cl_float * mass_dev,
							   cl_int * sort_dev,
							   cl_int * children_dev,
							   cl_int * maxdepth_dev,
							   cl_int * bottom_node_dev,
							   cl_float * boxsize_dev,
							   cl_int * num_nodes_dev, 
							   cl_int * num_bodies_dev,
							   cl_float * inv_opening_angle2_dev,  						               
							   cl_float * softening2_dev,
							   cl_int * children_local, 
							   cl_int * pos_local,
							   cl_int * node_local,
							   cl_float * dr_cutoff_local,
							   cl_float * nodex_local,
							   cl_float * nodey_local,
							   cl_float * nodez_local,
							   cl_float * nodem_local,
							   cl_int * wavefront_vote_local,
							   cl_int  wave_fronts_in_force_gravity_kernel
							){
  
  cl_int i, j, k, l,  node, depth, base, sbase, diff;
  cl_int maxdepth_local;
  cl_int block_size = wave_fronts_in_force_gravity_kernel*WAVEFRONT_SIZE;
  
  cl_float body_x [WAVEFRONT_SIZE];
  cl_float body_y [WAVEFRONT_SIZE];
  cl_float body_z [WAVEFRONT_SIZE];
  cl_float body_ax [WAVEFRONT_SIZE];
  cl_float body_ay [WAVEFRONT_SIZE];
  cl_float body_az [WAVEFRONT_SIZE];
  cl_float dx [WAVEFRONT_SIZE];
  cl_float dy [WAVEFRONT_SIZE];
  cl_float dz [WAVEFRONT_SIZE];
  cl_float temp_register [WAVEFRONT_SIZE];

  for (cl_int wavefront_leader_id = 0; wavefront_leader_id < block_size; wavefront_leader_id+=WAVEFRONT_SIZE){

    //prcl_intf("\n wavefront_leader_id = %d \n", wavefront_leader_id);

    if (wavefront_leader_id == 0){
      maxdepth_local = *maxdepth_dev;
      temp_register[0] = *boxsize_dev;
      //   prcl_intf("maxdepth_local = %.6f", maxdepth_local);
      //prcl_intf("temp_register[0] = %.6f", temp_register[0]);
      dr_cutoff_local[0] = temp_register[0] * temp_register[0] * (*inv_opening_angle2_dev);
      // prcl_intf("dr_cutoff_local[%d] = %.6f", 0, dr_cutoff_local[0]);
      for (i=1; i<maxdepth_local; i++){
	dr_cutoff_local[i] = dr_cutoff_local[i-1] * .25f;
	//prcl_intf("dr_cutoff_local[%d] = %.6f", i, dr_cutoff_local[i]);
      }
    }
      
    if (maxdepth_local <= MAX_DEPTH){
      base = wavefront_leader_id/WAVEFRONT_SIZE;
      j = base * MAX_DEPTH;
	
      for (cl_int wavefront_id = 0; wavefront_id < WAVEFRONT_SIZE; wavefront_id++){
	cl_int local_id = wavefront_leader_id + wavefront_id;
	sbase = base * WAVEFRONT_SIZE;
	diff = local_id - sbase;
	if(diff < MAX_DEPTH){
	  dr_cutoff_local[diff + j] = dr_cutoff_local[diff];
	}
      }
      
      for (k = wavefront_leader_id ; k < *num_bodies_dev; k += block_size){

	for (cl_int wavefront_id = 0; wavefront_id < WAVEFRONT_SIZE; wavefront_id++){
	  i = sort_dev[k + wavefront_id];
	  body_x[wavefront_id] = x_dev[i];
	  body_y[wavefront_id] = y_dev[i];
	  body_z[wavefront_id] = z_dev[i];

	  body_ax[wavefront_id] = 0.f;
	  body_ay[wavefront_id] = 0.f;
	  body_az[wavefront_id] = 0.f;
	}
	 
	depth = j;
	
	node_local[j] = *num_nodes_dev;
	pos_local[j] = 0;
	

	while (depth >= j){
	  while(pos_local[depth] < 8){
	    node = children_dev[node_local[depth]*8 + pos_local[depth]];
	    pos_local[depth]++;
	    children_local[base] = node;

	    if(node >= 0){
	      nodex_local[base] = x_dev[node];
	      nodey_local[base] = y_dev[node];
	      nodez_local[base] = z_dev[node];
	      nodem_local[base] = mass_dev[node];
	    }
	    node = children_local[base];

	    if (node >= 0){
	      
	      for (cl_int wavefront_id = 0; wavefront_id < WAVEFRONT_SIZE; wavefront_id++){
		dx[wavefront_id] = nodex_local[base] - body_x[wavefront_id];
		dy[wavefront_id] = nodey_local[base] - body_y[wavefront_id];
		dz[wavefront_id] = nodez_local[base] - body_z[wavefront_id];
		temp_register[wavefront_id] = dx[wavefront_id]*dx[wavefront_id];
		temp_register[wavefront_id] += dy[wavefront_id]*dy[wavefront_id];
		temp_register[wavefront_id] += dz[wavefront_id]*dz[wavefront_id];
		if(temp_register[wavefront_id] >= dr_cutoff_local[depth])
		  wavefront_vote_local[wavefront_leader_id + wavefront_id] = 1;
		else
		  wavefront_vote_local[wavefront_leader_id + wavefront_id] = 0;
	      }

	      for(l = 1; l < WAVEFRONT_SIZE; l++)
		wavefront_vote_local[wavefront_leader_id] += wavefront_vote_local[wavefront_leader_id+l];
	    
	      if((node < *num_bodies_dev) || wavefront_vote_local[wavefront_leader_id] >= WAVEFRONT_SIZE){
		for (cl_int wavefront_id = 0; wavefront_id < WAVEFRONT_SIZE; wavefront_id++){
		  if (node != sort_dev[k + wavefront_id]){
		    temp_register[wavefront_id] = 1.0f/sqrt(temp_register[wavefront_id] + *softening2_dev);
		    temp_register[wavefront_id] = nodem_local[base] * temp_register[wavefront_id]* temp_register[wavefront_id]* temp_register[wavefront_id];
		    body_ax[wavefront_id] += dx[wavefront_id] * temp_register[wavefront_id];
		    body_ay[wavefront_id] += dy[wavefront_id] * temp_register[wavefront_id];
		    body_az[wavefront_id] += dz[wavefront_id] * temp_register[wavefront_id];
		  }
		}
	      }
	      else {
		depth++;
		node_local[depth] = node;
		pos_local[depth] = 0;
	      }
	      
	    }
	    else {
	      if (j >= (depth - 1))
		depth = j;
	      else
		depth--;
	    }
	  }
	  depth--;
	}
	
	for (cl_int wavefront_id = 0; wavefront_id < WAVEFRONT_SIZE; wavefront_id++){
	  i = sort_dev[k + wavefront_id];
	  ax_dev[i] = body_ax[wavefront_id];
	  ay_dev[i] = body_ay[wavefront_id];
	  az_dev[i] = body_az[wavefront_id];
	}
      }
    }
  }
}
