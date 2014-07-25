
__kernel void cl_build_tree(
			    __global float* x_dev, 
			    __global float* y_dev,
			    __global float* z_dev,
			    __global float* mass_dev,
			    __global float* children_dev,
			    __global int bottom_dev,
			    __constant float boxsize_dev,
			    __constant float rootx_dev,
			    __constant float rooty_dev,
			    __constant float rootz_dev,
			    __constant float num_nodes_dev,
			    __constant float num_bodies_dev
			    )


{

  int i, j, k, parent_is_null, inc, child, n, cell, locked, patch;
  float x,y,z,r, px, py, pz;
  __local float root_radius, root_x, root_y, root_z;

  i = get_local_id(0);
  if (i == 0){
    root_radius = boxsize_dev/2.0f;
    root_x = x_dev[num_nodes_dev] = rootx_dev;
    root_y = y_dev[num_nodes_dev] = rooty_dev;
    root_z = z_dev[num_nodes_dev] = rootz_dev;
    mass_dev[num_nodes_dev] = -1.0f;
    //set root children to NULL
    for (k = 0; k < 8; k++)
      children_dev[num_nodes_dev*8 + k] = -1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  parent_is_null = 1;
  inc = get_global_size(0);
  i += get_group_id(0)*get_local_size(0); 

  while (i < num_bodies_dev){

    //insert new body
    if(parent_is_null){ 

      parent_is_null = 0;
      body_x = x_dev[i];
      body_y = y_dev[i];
      body_z = z_dev[i];
 
      //root node
      node = num_nodes_dev; 
      r = root_radius;
      j = 0;
      if (root_x < body_x) j = 1;
      if (root_y < booy_y) j += 2;
      if (root_z < body_z) j += 4;
    }
    
    child = children_dev[node*8 + j]; 

    //if child is not a leaf we
    //descend the tree untill we get 
    //to a NULL or filled leaf 
    while (child >= num_bodies_dev){
      node = child;
      r *= 0.5f;
      j = 0;
      if (x_dev[node] < body_x) j = 1;
      if (y_dev[node] < body_y) j += 2;
      if (z_dev[node] < body_z) j += 4;
      child = children_dev[node*8 + j];
    }

    //if child is not locked
    if (child != -2){ 
      locked = node*8 + j;
      if (child == atom_cmpxchg(&children_dev[locked], child, -2){ //try to lock child
	  
	  //if NULL, insert body
	  if(child == -1){
	    children_dev[locked] = i;
	  }
	  
	  //create new subtree by moving bottom_dev down one node
	  else {
	    patch = -1;
	    do {
	      cell = atomic_sub(&bottom_dev,1) - 1;

	      patch = max(patch,cell) ;
	      
	      x = (j & 1) * r;
	      y = ((j >> 1) & 1) * r;
	      z = ((j >> 2) & 1) * r;
	      r *= 0.5f;

	      mass_dev[cell] = -1.0f;
	      x = x_dev[cell] = x_dev[node] -r + x;
	      y = y_dev[cell] = y_dev[node] -r + y;
	      z = z_dev[cell] = z_dev[node] -r + z;

	      for (k = 0; k < 8; k++) 
		children_dev[cell*8 + k] = -1;

	      if (patch != cell)
		children_dev[node*8 + j] = cell;

	      j = 0;
	      if (x < x_dev[ch]) j = 1;
	      if (y < y_dev[ch]) j += 2;
	      if (z < z_dev[ch]) j += 4;
	      children_dev[cell*8+j] = ch;
	      
	      node = cell;
	      j = 0;
	      if (x < body_x) j = 1;
	      if (y < body_y) j += 2;
	      if (z < body_z) j += 4;
	      children_dev[cell*8 + j] = child;

	    } while ( child >= 0 ); 

	    children_dev[node*8 + j] = i;

	    // after mem_fence, all work items now see the added sub-tree
	    mem_fence(CLK_GLOBAL_MEM_FENCE); 

	  }
	  i += inc;
	  parent_is_null = 1;
	}
	}
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
