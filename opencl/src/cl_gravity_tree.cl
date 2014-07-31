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
								  __global int* children_dev,
								  __global int* maxdepth_dev,
								  __global int* bottom_node_dev,
								  __constant float* boxsize_dev,
								  __constant int* num_nodes_dev, 
								  __constant int* num_bodies_dev,
								  __constant float* inv_opening_angle2,  
								  __local *int children_local, 
								  __local *int pos_local,
								  __local *int node_local,
								  __local *float dq_local,
								  __local *float nodex_local,
								  __local *float nodey_local,
								  __local *float nodez_local,
								  __local *float nodem_local,
								  __local *int wavefront_vote_local
							     ){
  
  //POSSIBLE OPTIMIZATION: MAKE MAXDEPTH = WARPSIZE?
  int i, j, k, node, depth, base, sbase, diff, local_id;
  float body_x, body_y, body_z, body_ax, body_ay, body_az, dx, dy, dz, temp;
  __local int maxdepth_local;

  local_id = get_local_id(0);

  if (local_id == 0){
    maxdepth_local = maxdepth_dev;
    temp = boxsize_dev;
    dr_cutoff[0] = temp * temp * (*inv_opening_angle2);
    for (i  = 1; i < maxdepth_local; i++){
      dr_cutoff[i] = dr_cutoff[i-1] * .25f;
    }
    if (maxdepth_local > MAXDEPTH){
      *errd = -2;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (maxdepth <= MAXDEPTH){
    base = local_id / WAVEFRONT_SIZE;
    sbase = base * WAVEFRONT_SIZE;
    j = base * MAXDEPTH:
  
    diff = local_id - sbase;
    if (diff < MAXDEPTH){
      dr_cutoff[diff + j] = dr_cutoff[diff];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (k = local_id + get_group_id(0)*get_local_size(0); k < *num_bodies_dev; k += get_global_size(0)){

      i = sort_dev[k];
      body_x = x_dev[i];
      body_y = y_dev[i];
      body_z = z_dev[i];

      body_ax = 0.f;
      body_ay = 0.f;
      body_az = 0.f;
    
      depth = j;
      if (sbase == local_id){
	node_local[j] = num_nodes_dev;
	pos_local[j] = 0;
      }
      mem_fence(CLK_LOCAL_MEM_FENCE);

      while (depth >= j){
	while(pos[depth] < 8){
	  if(sbase == local_id){
	    node = children_dev[node_local[depth]*8 + pos_local[depth]];
	    pos_local[depth]++;
	    child_local[base] = node;
	    if (node >= 0){
	      wavefront_vote_local[base] = 0;
	      nodex_local[base] = x_dev[node]; 
	      nodey_local[base] = y_dev[node];
	      nodez_local[base] = z_dev[node];
	      nodem_local[base] = mass_dev[node];
	    }
	  }
	  mem_fence(CLK_LOCAL_MEM_FENCE);
	  
	  node = child_local[base];
	  if (node >= 0){
	    dx = nodex_local[base] - body_x;
	    dy = nodey_local[base] - body_y;
	    dz = nodez_local[base] - body_z;
	    temp = dx*dx + dy*dy + dz*dz;
	    if(temp >= dr_cutoff[depth])
	      atomic_inc(&wavefront_vote[base]);
	    if ((node < num_bodies_dev) || wavefront_vote[base] >= WAVEFRONT_SIZE){
	      if (node != i){
		temp = rsqrtf(temp + epssqd);
		temp = nm[base] * temp * temp * temp;
		body_ax += dx * temp;
		body_ay += dy * temp;
		body_az += dz * temp;
	      }
	    }
	    else{
	      depth++;
	      if(sbase == local_id){
		node_local[depth] = node;
		pos_local[depth] = 0;
	      }
	      mem_fence(CLK_LOCAL_MEM_FENCE);
	    }
	  }
	  else{
	    depth = max(j, depth - 1);
	  }
	}
	depth--;
      }

      ax_dev[i] = ax;
      ay_dev[i] = ay;
      az_dev[i] = az;
    }
  }
}
