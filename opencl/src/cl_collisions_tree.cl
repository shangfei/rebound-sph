float coefficient_of_restitution_bridges(float v){
	/* assumes v in units of [m/s] */
	float eps = 0.3f*pow(fabs(v)*100.f,-0.234f);
	if (eps>1) eps=1;
	if (eps<0) eps=0;
	return eps;

}

	  /* body2 = -1 -> no collision */
	  /* body2 = body1 -> same particle */
	  /* body2 < body1 -> another thread is already working on this collision or will be working on it (ASSUMPTION: only collisions between *pairs* of particles occur) */
	  /* body2 > body1 -> collision not resolved, fix this */
__kernel void cl_collisions_resolve(
				    __global float* x_dev,
				    __global float* y_dev,
				    __global float* z_dev,
				    __global float* vx_dev,
				    __global float* vy_dev,
				    __global float* vz_dev,
				    __global float* mass_dev,
				    __global float* rad_dev,
				    __global int* sort_dev,
				    __global int* collisions_dev,
				    __constant int* num_bodies_dev,
				    __constant float* minimum_collision_velocity_dev,
				    __constant float* OMEGA_dev,
				    __constant float* boxsize_dev,
				    __global float* t_dev,
				    __global float* error_dev
				     )

{

  //potential optimization: compaction
  //potential optimization: set if (body2 > body1) statement
  //near bottom of calculation to eliminate most of the thread
  //divergence by adding redundant calculations
  int i,body1,body2;
  int gbx_offset = *num_bodies_dev;
  int gby_offset = gbx_offset*3;
  int gbz_offset = gby_offset*3;
  int id = get_local_id(0) + get_group_id(0)*get_local_size(0);
  int inc = get_global_size(0);
  float shiftx, shifty, shiftz, shiftvx, shiftvy, shiftvz;

  for (i = id; i < *num_bodies_dev; i += inc){
    for (int gbx = -1; gbx <= 1; gbx++){
      for (int gby = -1; gby <= 1; gby++){
	for (int gbz = -1; gbz <= 1; gbz++){
	  body1 = sort_dev[i];
	  body2 = collisions_dev[body1 + gbx_offset*(gbx+1) + gby_offset*(gby+1) + gbz_offset*(gbz+1)];
	  if ( body2 > body1){
	    cl_boundaries_get_ghostbox(gbx,gby,gbz, &shiftx, &shifty, &shiftz, &shiftvx, &shiftvy, &shiftvz, *OMEGA_dev, *boxsize_dev, *boxsize_dev, *boxsize_dev, *t_dev);
  
	    float x21 = x_dev[body1] + shiftx - x_dev[body2];
	    float y21 = y_dev[body1] + shifty - y_dev[body2];
	    float z21 = z_dev[body1] + shiftz - z_dev[body2];
	    
	    float vx21 = vx_dev[body1] + shiftvx - vx_dev[body2];
	    float vy21 = vy_dev[body1] + shiftvy - vy_dev[body2];
	    float vz21 = vz_dev[body1] + shiftvz - vz_dev[body2];

	    float angle1 = atan2(z21, y21);
	    float sin_angle1 = sin(angle1);
	    float cos_angle1 = cos(angle1);
	    float vy21n = cos_angle1 * vy21 + sin_angle1 * vz21;
	    float y21n = cos_angle1 * y21 + sin_angle1 * z21;

	    float angle2 = atan2(y21n, x21);
	    float cos_angle2 = cos(angle2);
	    float sin_angle2 = sin(angle2);
	    float vx21nn = cos_angle2 * vx21 + sin_angle2 * vy21n;

	    float dvx2 = -(1.f + coefficient_of_restitution_bridges(vx21nn))*vx21nn;
	    float body1_r = rad_dev[body1];
	    float body2_r = rad_dev[body2];
	    float minr = (body1_r > body2_r)? body2_r:body1_r;
	    float maxr = (body1_r > body2_r)? body2_r:body1_r;
	    float mindv = minr*(*minimum_collision_velocity_dev);
	    float r = sqrt(x21*x21 + y21*y21 + z21*z21);

	    mindv *= 1.f-(r-maxr)/minr;
	    if (mindv > maxr*(*minimum_collision_velocity_dev))
	      mindv = maxr*(*minimum_collision_velocity_dev);
	    if (dvx2 < mindv)
	      dvx2 = mindv;
  
	    float dvx2n = cos_angle2 * dvx2;
	    float dvy2n = sin_angle2 * dvx2;
	    float dvy2nn = cos_angle1 * dvy2n;
	    float dvz2nn = sin_angle1 * dvy2n;
	    
	    float body1_mass = mass_dev[body1];
	    float body2_mass = mass_dev[body2];

	    float prefactor = body1_mass/(body1_mass + body2_mass);
	    vx_dev[body2] -= prefactor*dvx2n;
	    vy_dev[body2] -= prefactor*dvy2nn;
	    vz_dev[body2] -= prefactor*dvz2nn;

	    prefactor = body2_mass/(body1_mass + body2_mass);
	    vx_dev[body1] += prefactor*dvx2n;
	    vy_dev[body1] += prefactor*dvy2nn;
	    vz_dev[body1] += prefactor*dvz2nn;
	  }
	}
      }
    }
  }
}


__kernel void cl_collisions_search( 
				    __global float* x_dev, 
				    __global float* y_dev,
				    __global float* z_dev,
				    __global float* cellcenter_x_dev,
				    __global float* cellcenter_y_dev,
				    __global float* cellcenter_z_dev,
				    __global float* vx_dev,
				    __global float* vy_dev,
				    __global float* vz_dev,
				    __global float* mass_dev,
				    __global float* rad_dev,
				    __global int* sort_dev,
				    __global int* collisions_dev,
				    __global int* children_dev,
				    __global int* maxdepth_dev,
				    __constant float* boxsize_dev,
				    __constant int* num_nodes_dev, 
				    __constant int* num_bodies_dev,
				    __local int* children_local, 
				    __local int* pos_local,
				    __local int* node_local,
				    __local float* dr_cutoff_local,
				    __local float* nodex_local,
				    __local float* nodey_local,
				    __local float* nodez_local,
				    __local float* nodevx_local,
				    __local float* nodevy_local,
				    __local float* nodevz_local,
				    __local float* noderad_local,
				    __local int* wavefront_vote_local,
				    __constant float* collisions_max2_r_dev,
				    __constant float* OMEGA_dev,
				    __global float* t_dev
				    // __global float* error_dev
				    ){
  
  int i, j, k,l, node, depth, base, sbase, diff, local_id, gbx, gby, gbz;
  float body_x, body_y, body_z, body_vx, body_vy, body_vz, body_rad, dx, dy, dz, temp_register, shiftx, shifty, shiftz, shiftvx, shiftvy, shiftvz;
  __local int maxdepth_local;
    
  int gbx_offset = *num_bodies_dev;
  int gby_offset = gbx_offset*3;
  int gbz_offset = gby_offset*3;

  local_id = get_local_id(0);

  if (local_id == 0){
    maxdepth_local = *maxdepth_dev;
    dr_cutoff_local[0] = *boxsize_dev*0.86602540378443;
    for (i = 1; i < maxdepth_local; i++){
      dr_cutoff_local[i] = dr_cutoff_local[i-1] * .5f;
      dr_cutoff_local[i-1] += *collisions_max2_r_dev; 
    }
    dr_cutoff_local[maxdepth_local - 1] += *collisions_max2_r_dev;
#ifdef ERROR_CHECK
    if (maxdepth_local > MAXDEPTH){
    *error_dev = -2;
  }
#endif
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (maxdepth_local <= MAX_DEPTH){
    base = local_id / WAVEFRONT_SIZE;
    sbase = base * WAVEFRONT_SIZE;
    j = base * MAX_DEPTH;
  
    diff = local_id - sbase;
    if (diff < MAX_DEPTH){
      dr_cutoff_local[diff + j] = dr_cutoff_local[diff];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //potential optimization: replace these with stored register variables
    for (k = local_id + get_group_id(0)*get_local_size(0); k < *num_bodies_dev; k += get_global_size(0)){
  
      i = sort_dev[k];
      body_x = x_dev[i];
      body_y = y_dev[i];
      body_z = z_dev[i];
      body_vx = vx_dev[i];
      body_vy = vy_dev[i];
      body_vz = vz_dev[i];
      body_rad = rad_dev[i];

      for (gbx = -1; gbx <= 1; gbx++){
	for (gby = -1; gby <= 1; gby++){
	  for (gbz = -1; gbz <= 1; gbz++){

	    //send shifts to function as pointers, get rid of struct
	    cl_boundaries_get_ghostbox(gbx,gby,gbz,&shiftx,&shifty,&shiftz,&shiftvx,&shiftvy,&shiftvz, *OMEGA_dev,*boxsize_dev, *boxsize_dev, *boxsize_dev, *t_dev);
	    /* shiftvx = 0.f; */
	    /* shiftvy = -1.5f*(float)gbx*(*OMEGA_dev)*(*boxsize_dev); */
	    /* shiftvz = 0.f; */
	    /* float shift = (gbx == 0) ? 0.f : -fmod(shiftvy*(*t_dev) - ((gbx>0) - (gbx<0))*(*boxsize_dev)/2.f, *boxsize_dev) -  ((gbx>0) - (gbx<0))*(*boxsize_dev)/2.f; */
	    /* shiftx = *boxsize_dev*(float)gbx; */
	    /* shifty = *boxsize_dev*(float)gby-shift; */
	    /* shiftz = *boxsize_dev*(float)gbz; */
	 
	    depth = j;
	    //first thread in wavefront leads
	    if (sbase == local_id){
	      node_local[j] = *num_nodes_dev;
	      pos_local[j] = 0;
	    }
	    mem_fence(CLK_LOCAL_MEM_FENCE);

	    //initialize collisions array
	    collisions_dev[i + gbx_offset*(gbx+1) + gby_offset*(gby+1) + gbz_offset*(gbz+1)] = -1; 

	    while (depth >= j){
	      while(pos_local[depth] < 8){
		//first thread in wavefront leads
		if(sbase == local_id){
		  node = children_dev[node_local[depth]*8 + pos_local[depth]];
		  pos_local[depth]++;
		  children_local[base] = node;
		  if (node >= 0){
		    //MIGHT CHANGE TO THIS TO NODE >= 0 && NODE < NUM_BODIES?
		    if (node < *num_bodies_dev){
		      nodex_local[base] = x_dev[node];
		      nodey_local[base] = y_dev[node];
		      nodez_local[base] = z_dev[node];
		      nodevx_local[base] = vx_dev[node];
		      nodevy_local[base] = vy_dev[node];
		      nodevz_local[base] = vz_dev[node];
		      noderad_local[base] = rad_dev[node];
		    }
		    
		    else{
		      nodex_local[base] = cellcenter_x_dev[node];
		      nodey_local[base] = cellcenter_y_dev[node];
		      nodez_local[base] = cellcenter_z_dev[node];
		    }
		  }
		}
		mem_fence(CLK_LOCAL_MEM_FENCE);

		//each wavefront member grabs the node the wavefront leader put in local memory
		node = children_local[base];

		if (node >= 0){

		  dx = (body_x + shiftx) - nodex_local[base];
		  dy = (body_y + shifty) - nodey_local[base];
		  dz = (body_z + shiftz) - nodez_local[base];
		  temp_register = dx*dx + dy*dy + dz*dz;
 		  // if it's a leaf cell

		  if (node < *num_bodies_dev) 
		    {
		      //if the node isn't the body check for a collision 
		      if (node != i){
			float rp = body_rad + noderad_local[base];
			float dvx = (body_vx + shiftvx) - nodevx_local[base]; 
			float dvy = (body_vy + shiftvy) - nodevy_local[base];
			float dvz = (body_vz + shiftvz) - nodevz_local[base];        	
		 
			if ( temp_register <=  rp*rp && dvx*dx + dvy*dy + dvz*dz < 0){	  
			  collisions_dev[i + gbx_offset*(gbx+1) + gby_offset*(gby+1) + gbz_offset*(gbz+1)] = node; 

			}
		      }
		    }
		  
		  else{
		    float rp = body_rad + dr_cutoff_local[depth];
		    wavefront_vote_local[local_id] = (temp_register >= rp*rp) ? 1 : 0;
		    
		    if (local_id == sbase)
		      for(l = 1; l < WAVEFRONT_SIZE; l++)
			wavefront_vote_local[sbase] += wavefront_vote_local[sbase + l];
		    //this blocks the warp from moving forward since they must wait for their leader (local_id = sbase)
		    mem_fence(CLK_LOCAL_MEM_FENCE);
		    
		    //the warp votes whether or not to descend. If one member of the warp votes descend, the
		    //warp descends
		    if (wavefront_vote_local[sbase] < WAVEFRONT_SIZE){
		      depth++;
		      if(sbase == local_id){
			node_local[depth] = node;
			pos_local[depth] = 0;
		      }
		      mem_fence(CLK_LOCAL_MEM_FENCE);
		    }
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
    }
  }
}
