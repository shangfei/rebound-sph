#include "cl_gpu_defns.h"
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

__kernel void cl_gravity_calculate_acceleration_for_particle(
							     __global float* x_dev, 
							     __global float* y_dev,
							     __global float* z_dev,
							     __global float* ax_dev,
							     __global float* ay_dev,
							     __global float* az_dev,
							     __global float* mass_dev,
							     __global int* sort_dev,
							     __global int* children_dev,
							     __global int* maxdepth_dev,
							     __global float* t_dev,
							     __constant float* OMEGA_dev,
							     __constant float* boxsize_dev,
							     __constant int* num_nodes_dev, 
							     __constant int* num_bodies_dev,
							     __constant float* inv_opening_angle2_dev,  							
							     __constant float* softening2_dev,
							     __constant float* G_dev,
							     __local int* children_local, 
							     __local int* pos_local,
							     __local int* node_local,
							     __local float* dr2_cutoff_local,
							     __local float* nodex_local,
							     __local float* nodey_local,
							     __local float* nodez_local,
							     __local float* nodem_local,
							     __local int* wavefront_vote_local
							     //__global float* error_dev
							     ){
  

  int i, j, k,l, node, depth, base, sbase, diff, local_id;
  float body_x, body_y, body_z, body_ax, body_ay, body_az, dx, dy, dz, temp_register;
  float shiftx, shifty, shiftz;
  __local int maxdepth_local;
  __local float t_local;

#ifdef KAHAN_SUMMATION
  // accumulators for Kahan summation of phase
  float kahanC_ax = 0.f; 
  float kahanC_ay = 0.f; 
  float kahanC_az = 0.f; 
  float kahanT = 0.f;
  float kahanY = 0.f;
#endif
  
  local_id = get_local_id(0);

  if (local_id == 0){
    maxdepth_local = *maxdepth_dev;
    temp_register = *boxsize_dev;
    t_local = *t_dev;
    dr2_cutoff_local[0] = temp_register * temp_register * (*inv_opening_angle2_dev);
    for (i  = 1; i < maxdepth_local; i++)
      dr2_cutoff_local[i] = dr2_cutoff_local[i-1] * .25f;
#ifdef ERROR_CHECK
    if (maxdepth_local > MAXDEPTH){
      *error_dev = -2;
    }
#endif
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float t = t_local;

  if (maxdepth_local <= MAX_DEPTH){
    base = local_id / WAVEFRONT_SIZE;
    sbase = base * WAVEFRONT_SIZE;
    j = base * MAX_DEPTH;
  
    diff = local_id - sbase;
    if (diff < MAX_DEPTH){
      dr2_cutoff_local[diff + j] = dr2_cutoff_local[diff];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //potential optomization: replace these with stored register variables
    for (k = local_id + get_group_id(0)*get_local_size(0); k < *num_bodies_dev; k += get_global_size(0)){

      i = sort_dev[k];
      body_x = x_dev[i];
      body_y = y_dev[i];
      body_z = z_dev[i];

      body_ax = 0.f;
      body_ay = 0.f;
      body_az = 0.f;

      for (int gbx = -1; gbx <= 1; gbx++){
      	for (int gby = -1; gby <= 1; gby++){
      	  for (int gbz = -1; gbz <= 1; gbz++){

      	    cl_boundaries_get_ghostbox(gbx,gby,gbz,&shiftx,&shifty,&shiftz,&temp_register,&temp_register,&temp_register, *OMEGA_dev, *boxsize_dev, *boxsize_dev, *boxsize_dev, t);
    
	    depth = j;
	    //first thread in wavefront leads
	    if (sbase == local_id){
	      node_local[j] = *num_nodes_dev;
	      pos_local[j] = 0;
	    }
	    mem_fence(CLK_LOCAL_MEM_FENCE);

	    while (depth >= j){
	      while(pos_local[depth] < 8){
		//first thread in wavefront leads
		if(sbase == local_id){
		  node = children_dev[node_local[depth]*8 + pos_local[depth]];
		  pos_local[depth]++;
		  children_local[base] = node;
		  if (node >= 0){
		    nodex_local[base] = x_dev[node] + shiftx;
		    nodey_local[base] = y_dev[node] + shifty;
		    nodez_local[base] = z_dev[node] + shiftz;
		    nodem_local[base] = mass_dev[node];
		  }
		}
		mem_fence(CLK_LOCAL_MEM_FENCE);
		node = children_local[base];
		if (node >= 0){
		  dx = nodex_local[base] - body_x;
		  dy = nodey_local[base] - body_y;
		  dz = nodez_local[base] - body_z;
		  temp_register = dx*dx + dy*dy + dz*dz;

		  wavefront_vote_local[local_id] = (temp_register >= dr2_cutoff_local[depth]) ? 1 : 0;

		  if (local_id == sbase)
		    for(l = 1; l < WAVEFRONT_SIZE; l++)
		      wavefront_vote_local[sbase] += wavefront_vote_local[sbase + l];
		  mem_fence(CLK_LOCAL_MEM_FENCE);
	    
		  //the node is either a body or the wavefront votes that the node cell is too far away
		  //optimization: maybe switch order of these if conditions, b/c local mem faster than const

		  if ((node < *num_bodies_dev) || wavefront_vote_local[sbase] >= WAVEFRONT_SIZE){
		    //if the node isn't the body we are computing the acc for
		    if (node != i){
		      temp_register = rsqrt(temp_register + *softening2_dev);
		      temp_register = *G_dev * nodem_local[base] * temp_register * temp_register * temp_register;
#ifdef KAHAN_SUMMATION
		      kahanY = dx * temp_register - kahanC_ax;
		      kahanT = body_ax + kahanY;
		      kahanC_ax = (kahanT - body_ax) - kahanY;
		      body_ax = kahanT;
	
		      kahanY = dy * temp_register - kahanC_ay;
		      kahanT = body_ay + kahanY;
		      kahanC_ay = (kahanT - body_ay) - kahanY;
		      body_ay = kahanT;

		      kahanY = dz * temp_register - kahanC_az;
		      kahanT = body_az + kahanY;
		      kahanC_az = (kahanT - body_az) - kahanY;
		      body_az = kahanT;
#else
		      body_ax += dx * temp_register;
		      body_ay += dy * temp_register;
		      body_az += dz * temp_register;		
#endif
		    }
		  }

		  //descend into child cell
		  else{
		    depth++;
		    if(sbase == local_id){
		      node_local[depth] = node;
		      pos_local[depth] = 0;
		    }
		    mem_fence(CLK_LOCAL_MEM_FENCE);
		  }
		}

		//if child is null then remaining children of this node is null so move back up the tree
		else{
		  depth = max(j, depth - 1);
		}
	      }
	      depth--;
	    }
	  }
	}
      }
      ax_dev[i] = body_ax;
      ay_dev[i] = body_ay;
      az_dev[i] = body_az;
    }
  }
}
