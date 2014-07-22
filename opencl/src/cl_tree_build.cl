__kernel void cl_build_tree(
			    __global float* posx_dev, 
			    __global float* posy_dev,
			    __global float* posz_dev,
			    __global float* mass_dev,
			    __global float* tree_dev,
			    __constant float radius_dev,
			    __constant float num_nodes_dev,
			    __constant float num_bodies_dev
			    ){


  int i, j, k, depth, localmaxdepth, parent_is_null, inc, child, n, cell, locked, patch;
  float x,y,z,r, px, py, pz;
  __local float root_radius, root_x, root_y, root_z;

  i = get_local_id(0);
  if (i == 0){
    root_radius = radius_dev;
    root_x = posx_dev[num_nodes_dev];
    root_y = posy_dev[num_nodes_dev];
    root_z = posz_dev[num_nodes_dev];
  }
  
  localmaxdepth = 1;
  parent_is_null = 1;
  inc = get_global_size(0);
  i += get_group_id(0)*get_local_size(0); //can we do this in one call instead of two?

  while (i < num_bodies_dev){
    if(parent_is_null){ // new body, start at root 
      parent_is_null = 0;
      body_x = posx_dev[i];
      body_y = posy_dev[i];
      body_z = posz_dev[i];
      node = num_nodes_dev; // root node id
      r = root_radius;
      j = 0;
      if (root_x < body_x) j = 1;
      if (root_y < booy_y) j += 2;
      if (root_z < body_z) j += 4;
    }
    
    child = tree_dev[node*8 + j]; 

    while (child >= num_bodies_dev){
      node = child;
      depth++;
      r *= 0.5f;
      j = 0;
      if (posx_dev[node] < body_x) j = 1;
      if (posy_dev[node] < body_y) j += 2;
      if (posz_dev[node] < body_z) j += 4;
      child = tree_dev[node*8 + j];
    }

    
    if (child != -2){
      locked = node*8 + j;
      if (ch == atom_cmpxchg(&tree_dev[locked], child, -2){
	  if(child == -1){
	    tree_dev[lcoked] = i;
	  }
	  else {
	    patch = -1;
	    do {
	      depth++;
	      cell = atomic_sub(&bottomd_,1) - 1;
	      if (cell <= num_bodies_dev){
		*errd = 1;
		bottomd = nnodesd;
	      }

	      patch = max(patch,cell); // figure out max function
	      
	      x = (j & 1) * r;
	      y = ((j >> 1) & 1) * r;
	      z = ((j >> 2) & 1) * r;
	      r *= 0.5f;

	      mass_dev[cell] = -1.0f;
	      x = posxd[cell] = posxd[node] -r + x;
	      y = posyd[cell] = posyd[node] -r + y;
	      z = poszd[cell] = poszd[node] -r + z;

	      for (k = 0; k < 8; k++) tree_dev[cell*8 + k] = -1;

	      if (patch != cell){
		tree_dev[node*8 + j] = cell;
	      }

	      j = 0;
	      if (x < posx_dev[ch]) j = 1;
	      if (y < posy_dev[ch]) j += 2;
	      if (z < posz_dev[ch]) j += 4;
	      child_dev[cell*8+j] = ch;
	      
	      node = cell;
	      j = 0;
	      if (x < body_x) j = 1;
	      if (y < body_y) j += 2;
	      if (z < body_z) j += 4;
	      tree_dev[cell*8 + j] = child;

	      n = cell;
	      j=0;
	      if ( x < body_x ) j = 1;
	      if ( y < body_y ) j += 2;
	      if ( z < body_z ) j += 4;

	      child = tree_dev[node*8 + j];
	    } while ( child >= 0 ); 

	    childd[n*8 + j] = i;
	    mem_fence(); // research this function

	  }

	  localmaxdepth = max(depth, localmaxdepth);
	  i += inc;
	  parent_is_null = 1;
	}
	}
      __syncthreads();
    }
    atomicMax(&maxdepthd, localmaxdepth);
  }
