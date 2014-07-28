#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics: enable

__kernel void cl_build_tree(
			    __global float* x_dev, 
			    __global float* y_dev,
			    __global float* z_dev,
			    __global float* mass_dev,
			    __global int* children_dev,
			    __global int* bottom_node_dev,
			    __constant float* boxsize_dev,
			    __constant float* rootx_dev,
			    __constant float* rooty_dev,
			    __constant float* rootz_dev,
			    __constant int* num_nodes_dev, 
			    __constant int* num_bodies_dev
			    )
{
  //POTENTIAL OPTIMIZATION: load num_nodes, num_bodies into registers before use

  int i, j, k, parent_is_null, inc, child, node, cell, locked, patch;
  float body_x,body_y,body_z,r, cell_x, cell_y, cell_z;
  __local float root_cell_radius, root_cell_x, root_cell_y, root_cell_z;

  i = get_local_id(0);
  if (i == 0){
    root_cell_radius = *boxsize_dev/2.0f;
    root_cell_x = x_dev[*num_nodes_dev] = *rootx_dev;
    root_cell_y = y_dev[*num_nodes_dev] = *rooty_dev;
    root_cell_z = z_dev[*num_nodes_dev] = *rootz_dev;
    mass_dev[*num_nodes_dev] = -1.0f;
    *bottom_node_dev = *num_nodes_dev;
    //set root children to NULL
    for (k = 0; k < 8; k++)
      children_dev[*num_nodes_dev*8 + k] = -1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  parent_is_null = 1;
  inc = get_global_size(0);
  i += get_group_id(0)*get_local_size(0);

  while (i < *num_bodies_dev){
    //insert new body
    if(parent_is_null == 1){

      parent_is_null = 0;
      body_x = x_dev[i];
      body_y = y_dev[i];
      body_z = z_dev[i];
 
      //root node
      node = *num_nodes_dev;
      r = root_cell_radius;
      j = 0;
      if (root_cell_x < body_x) j = 1;
      if (root_cell_y < body_y) j += 2;
      if (root_cell_z < body_z) j += 4;
    }
    
    child = children_dev[node*8 + j];

    //if child is not a leaf we
    //descend the tree untill we get
    //to a NULL or filled leaf
    while (child >= *num_bodies_dev){
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
      if (child == atom_cmpxchg(&children_dev[locked], child, -2)){ //try to lock child
	  
  	  //if NULL, insert body
  	  if(child == -1){
  	    children_dev[locked] = i;
  	  }
	  
  	  //create new subtree by moving *bottom_node_dev down one node
  	  else {
  	    patch = -1;
  	    do {
  	      cell = atomic_sub(bottom_node_dev,1) - 1;

  	      patch = max(patch,cell) ;
	      
  	      cell_x = (j & 1) * r;
  	      cell_y = ((j >> 1) & 1) * r;
  	      cell_z = ((j >> 2) & 1) * r;
  	      r *= 0.5f;

  	      mass_dev[cell] = -1.0f;
  	      cell_x = x_dev[cell] = x_dev[node] -r + cell_x;
  	      cell_y = y_dev[cell] = y_dev[node] -r + cell_y;
  	      cell_z = z_dev[cell] = z_dev[node] -r + cell_z;

  	      for (k = 0; k < 8; k++)
  		children_dev[cell*8 + k] = -1;

  	      if (patch != cell)
  		children_dev[node*8 + j] = cell;

  	      j = 0;
  	      if (cell_x < x_dev[child]) j = 1;
  	      if (cell_y < y_dev[child]) j += 2;
  	      if (cell_z < z_dev[child]) j += 4;
  	      children_dev[cell*8+j] = child;
	      
  	      node = cell;
  	      j = 0;
  	      if (cell_x < body_x) j = 1;
  	      if (cell_y < body_y) j += 2;
  	      if (cell_z < body_z) j += 4;
  	      child = children_dev[node*8 + j];

  	    } while ( child >= 0 );

  	    children_dev[node*8 + j] = i;

  	    // after mem_fence, all work items now see the added sub-tree
  	    mem_fence(CLK_GLOBAL_MEM_FENCE);
	    children_dev[locked] = patch;
  	  }
  	  i += inc;
  	  parent_is_null = 1;
  	}
  	}
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
