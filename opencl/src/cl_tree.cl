#include "cl_gpu_defns.h"
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

__kernel void cl_tree_add_particles_to_tree(
			    __global float* x_dev, 
			    __global float* y_dev,
			    __global float* z_dev,
			    __global float* mass_dev,
			    __global float* start_dev,
			    __global int* children_dev,
			    __global int* maxdepth_dev,
			    __global int* bottom_node_dev,
			    __constant float* boxsize_dev,
			    __constant float* rootx_dev,
			    __constant float* rooty_dev,
			    __constant float* rootz_dev,
			    __constant int* num_nodes_dev, 
			    __constant int* num_bodies_dev
			    )
{
  //POTENTIAL SIMPLE OPTIMIZATION: load num_nodes, num_bodies into registers before use

  int i, j, k, parent_is_null, inc, child, node, cell, locked, patch, maxdepth_thread, depth;
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
    *maxdepth_dev = 1;
    //set root children to NULL
    for (k = 0; k < 8; k++)
      children_dev[*num_nodes_dev*8 + k] = -1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  maxdepth_thread = 1;
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
      depth = 1;
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
      depth++;
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
	      depth++;
  	      cell = atomic_sub(bottom_node_dev,1) - 1;

  	      patch = max(patch,cell) ;
	      
  	      cell_x = (j & 1) * r;
  	      cell_y = ((j >> 1) & 1) * r;
  	      cell_z = ((j >> 2) & 1) * r;
  	      r *= 0.5f;

  	      mass_dev[cell] = -1.0f;
	      start_dev[cell] = -1;
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
	  maxdepth_thread = max(depth, maxdepth_thread);
  	  i += inc;
  	  parent_is_null = 1;
  	}
  	}
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  atomic_max(maxdepth_dev, maxdepth_thread);
  }


__kernel void cl_tree_update_tree_gravity_data(
					           __global float* x_dev, 
						   __global float* y_dev,
						   __global float* z_dev,
						   __global float* mass_dev,
						   __global int* children_dev,
						   __global int* count_dev,
						   __global int* bottom_node_dev,
						   __constant int* num_nodes_dev, 
						   __constant int* num_bodies_dev,
						   __local int* children_local
					         )
{

  //POTENTIAL SIMPLE OPTIMIZATION: replace instances of get_local_id(0) with a register variable thread_id

  int i, num_processed, k, child, inc, num_notcalculated, count, num_group_threads, local_id;
  float child_mass, cell_mass, cell_x, cell_y, cell_z;
  __local int bottom_node;
  
  local_id = get_local_id(0);

  if (local_id == 0){
    bottom_node = *bottom_node_dev;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  
  inc = get_global_size(0);
  num_group_threads = get_local_size(0);
  num_notcalculated = 0;

  k = (bottom_node & (-WARP_SIZE)) + local_id + get_group_id(0)*num_group_threads;
  if (k < bottom_node) 
    k += inc;

  while (k <= *num_nodes_dev){

    if (num_notcalculated == 0){
      cell_mass = 0.f;
      cell_x = 0.f;
      cell_y = 0.f;
      cell_z = 0.f;
      count = 0;
      num_processed = 0;

      for (i = 0; i < 8; i++){
  	child = children_dev[k*8 + i];
  	if (child >= 0){
  	  if (i != num_processed){
  	    //move child to the front
  	    children_dev[k*8+i] = -1;
  	    children_dev[k*8+num_processed] = child;
  	  }
  	  children_local[num_notcalculated*num_group_threads + local_id] = child;
  	  child_mass = mass_dev[child];
  	  num_notcalculated++;

  	  if(child_mass >= 0.f){
  	    //cell is ready to be calculated
  	    num_notcalculated--;
  	    if (child >= *num_bodies_dev){
  	      count += count_dev[child] - 1; //subtract one
  	    }
  	    cell_mass += child_mass;
  	    cell_x += x_dev[child]*child_mass;
  	    cell_y += y_dev[child]*child_mass;
  	    cell_z += z_dev[child]*child_mass;
  	  }
  	  num_processed++;
  	}
      }
      count += num_processed;
    }
    
    //some child cells have more than one body
    //and so we need to descend the tree
    if (num_notcalculated != 0){
      do {
  	child = children_local[(num_notcalculated - 1)*num_group_threads + get_local_id(0)];
  	child_mass = mass_dev[child];
  	if (child_mass >= 0.f){
  	  num_notcalculated--;
  	  if (child >= *num_bodies_dev){
  	    count += count_dev[child] - 1; //we subtract one b/c num_processed counts the cell
  	  }
  	  cell_mass += child_mass;
  	  cell_x += x_dev[child] * child_mass;
  	  cell_y += y_dev[child] * child_mass;
  	  cell_z += z_dev[child] * child_mass;
  	}
      } while ( (child_mass >= 0.f) && (num_notcalculated != 0) );
    }
    
    if (num_notcalculated == 0){
      count_dev[k] = count;
      //child_mass is used as a temporary storage device here
      //for the inverse mass
      child_mass = 1.0f/cell_mass;
      x_dev[k]=cell_x*child_mass;
      y_dev[k]=cell_y*child_mass;
      z_dev[k]=cell_z*child_mass;
      //should this mem_fence be before or after mass_dev[k] =?
      mem_fence(CLK_GLOBAL_MEM_FENCE);
      mass_dev[k] = cell_mass;
      k += inc;
    }
  }
}
  
/* __kernel void cl_tree_sort_particles(){ */
  
/*   int i,k, child, dec, start, bottom; */
/*   __local int bottoms; */
  
/*   //Optimization: get rid of this? */
/*   if (get_local_id(0) == 0){ */
/*     bottoms = *bottom_dev; */
/*   } */
/*   barrier(CLK_LOCAL_MEM_FENCE); */
/*   bottom = bottoms; */
  
/*   dec = get_global_size(0); */
/*   k = num_nodes_dev + 1 - dec + get_local_id(0) + get_group_id(0)*get_local_size(0); */

/*   while (k >= bottom){ */
/*     start = start_dev[k]; */
/*     if (start >= 0){ */
      
/*       for (i = 0; i < 8; i++){ */
/* 	child = children_dev[k*8 + i]; */
/* 	if (child >= num_bodies_dev){ */
/* 	  start_dev[child] = start; */
/* 	  start += count_dev[child]; */
/* 	} */
/* 	else { */
/* 	  sort_dev[start] = child; */
/* 	  start++l */
/* 	} */
/*       } */
/*       k -= dec; */
/*     } */
/*   } */
/* } */
    

