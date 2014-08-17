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
#include "cpu_collisions_search.h"


void boundaries_get_ghostbox(
			     int i, 
			     int j, 
			     int k,
			     float *shiftx,
			     float *shifty,
			     float *shiftz,
			     float *shiftvx,
			     float *shiftvy,
			     float *shiftvz,
			     float *OMEGA,
			     float *boxsize_x,
			     float *boxsize_y,
			     float *boxsize_z,
			     float *t
			     ){
  *shiftvx = 0.f;
  *shiftvy = -1.5f*(float)i*(*OMEGA)*(*boxsize_x);
  *shiftvz = 0.f;
  float shift = (i == 0) ? 0.f : -fmod( (*shiftvy)*(*t) - ((i>0) - (i<0))* (*boxsize_y)/2.f, (*boxsize_y)) -  ((i>0) - (i<0))*(*boxsize_y)/2.f;
  *shiftx = (*boxsize_x)*(float)i;
  *shifty = (*boxsize_y)*(float)j-shift;
  *shiftz = (*boxsize_z)*(float)k;
}

/* void collisions_resolve(){ */
/* 	// Loop over all collisions previously found in collisions_search(). */
/* 	for (int i=0;i<collisions_N;i++){ */
/* 		// Resolve collision */
/* 		collision_resolve(collisions[i]); */
/* 	} */
/* 	// Mark all collisions as resolved. */
/* 	collisions_N=0; */
/* } */

void cpu_collisions_search(	
			   cl_float * x_host, 
			   cl_float * y_host,
			   cl_float * z_host,
			   cl_float * rad_host,
			   cl_float * vx_host,
			   cl_float * vy_host,
			   cl_float * vz_host,
			   cl_int * collisions_host,
			   cl_int * num_bodies_host,
			   cl_int * nghostx,
			   cl_int * nghosty,
			   cl_int * nghostz,
			   cl_float *boxsize_host,
			   cl_float *OMEGA_host,
			   cl_float *t_host
				){
    
  float shiftx,shifty,shiftz,shiftvx,shiftvy,shiftvz;
  float body1_x, body1_y, body1_z, body1_vx, body1_vy, body1_vz;
  float body2_x, body2_y, body2_z, body2_vx, body2_vy, body2_vz;
  float sr, r2, dx, dy, dz, dvx, dvy, dvz;

  int gbx_offset = *num_bodies_host;
  int gby_offset = gbx_offset*(*nghostx + 1);
  int gbz_offset = gby_offset*(*nghosty + 1);
  int nghostxcol = (*nghostx>1?1:*nghostx);
  int nghostycol = (*nghosty>1?1:*nghosty);
  int nghostzcol = (*nghostz>1?1:*nghostz);

  for (int gbx=-nghostxcol; gbx<=nghostxcol; gbx++){
    for (int gby=-nghostycol; gby<=nghostycol; gby++){
      for (int gbz=-nghostzcol; gbz<=nghostzcol; gbz++){
	// Loop over all particles

	boundaries_get_ghostbox(gbx,gby,gbz,&shiftx,&shifty,&shiftz,&shiftvx,&shiftvy,&shiftvz, OMEGA_host,boxsize_host, boxsize_host, boxsize_host, t_host);
	for (int i=0;i<*num_bodies_host;i++){

	  collisions_host[i + gbx_offset*(gbx+1) + gby_offset*(gby+1) + gbz_offset*(gbz+1)] = -1;
	  body1_x = x_host[i] + shiftx;
	  body1_y = y_host[i] + shifty;
	  body1_z = z_host[i] + shiftz;
	  body1_vx = vx_host[i] + shiftvx;
	  body1_vy = vy_host[i] + shiftvy;
	  body1_vz = vz_host[i] + shiftvz;
	  // Loop over all particles again
	  for (int j=0;j<*num_bodies_host;j++){
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
	    collisions_host[i + gbx_offset*(gbx+1) + gby_offset*(gby+1) + gbz_offset*(gbz+1)] = j; 
	    if (i == 201 && j == 55 && gbx == 1 && gby == 0 && gbz == 1){
	      printf("THERE IS A PROBLEM\n\n");
	    }
	  }
	}
      }
    }
  }
}



void cpu_collisions_search_2(	
			     cl_float * x_dev, 
			     cl_float * y_dev,
			     cl_float * z_dev,
			     cl_float* cellcenter_x_dev,
			     cl_float* cellcenter_y_dev,
			     cl_float* cellcenter_z_dev,
			     cl_float * vx_dev,
			     cl_float * vy_dev,
			     cl_float * vz_dev,
			     cl_float * mass_dev,
			     cl_float* rad_dev,
			     cl_int* sort_dev,
			     cl_int* collisions_dev,
			     cl_int* children_dev,
			     cl_int* maxdepth_dev,
			     cl_float* boxsize_dev,
			     cl_int* num_nodes_dev, 
			     cl_int* num_bodies_dev,
			     cl_int* children_local, 
			     cl_int* pos_local,
			     cl_int* node_local,
			     cl_float* dr_cutoff_local,
			     cl_float* nodex_local,
			     cl_float* nodey_local,
			     cl_float* nodez_local,
			     cl_float* nodevx_local,
			     cl_float* nodevy_local,
			     cl_float* nodevz_local,
			     cl_float* nodem_local,
			     cl_float* noderad_local,
			     cl_int* wavefront_vote_local,
			     cl_float* collisions_max2_r_dev,
			     cl_float* OMEGA_dev,
			     cl_float* t_dev,
			     cl_int wave_fronts_in_force_gravity_kernel
			     ){
  
  cl_int i, j, k, l,  node, depth, base, sbase, diff, gbx, gby, gbz;
  cl_int maxdepth_local;
  cl_int block_size = wave_fronts_in_force_gravity_kernel*WAVEFRONT_SIZE;
;
  cl_float shiftx, shifty, shiftz, shiftvx, shiftvy, shiftvz;
  cl_int gbx_offset = *num_bodies_dev;
  cl_int gby_offset = gbx_offset*3;
  cl_int gbz_offset = gby_offset*3;

  cl_float body_x [WAVEFRONT_SIZE];
  cl_float body_y [WAVEFRONT_SIZE];
  cl_float body_z [WAVEFRONT_SIZE];
  cl_float body_vx [WAVEFRONT_SIZE];
  cl_float body_vy [WAVEFRONT_SIZE];
  cl_float body_vz [WAVEFRONT_SIZE];
  cl_float body_rad [WAVEFRONT_SIZE];
  cl_float dx [WAVEFRONT_SIZE];
  cl_float dy [WAVEFRONT_SIZE];
  cl_float dz [WAVEFRONT_SIZE];
  cl_float dvx [WAVEFRONT_SIZE];
  cl_float dvy [WAVEFRONT_SIZE];
  cl_float dvz [WAVEFRONT_SIZE];
  cl_float temp_register [WAVEFRONT_SIZE];

  for (cl_int wavefront_leader_id = 0; wavefront_leader_id < block_size; wavefront_leader_id+=WAVEFRONT_SIZE){

    //printf("\n wavefront_leader_id = %d \n", wavefront_leader_id);


    if (wavefront_leader_id == 0){
      printf("\n*maxdepth_dev = %d\n", (int)*maxdepth_dev);
      maxdepth_local = *maxdepth_dev;
      dr_cutoff_local[0] = *boxsize_dev*0.86602540378443;
      for (i = 1; i < maxdepth_local; i++){
	dr_cutoff_local[i] = dr_cutoff_local[i-1] * .5f;
	dr_cutoff_local[i-1] += *collisions_max2_r_dev; 
      }
      dr_cutoff_local[maxdepth_local - 1] += *collisions_max2_r_dev;
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
	  body_vx[wavefront_id] = vx_dev[i];
	  body_vy[wavefront_id] = vy_dev[i];
	  body_vz[wavefront_id] = vz_dev[i];
	  body_rad[wavefront_id] = rad_dev[i];
	}

	for (gbx = -1; gbx <= 1; gbx++){
	  for (gby = -1; gby <= 1; gby++){
	    for (gbz = -1; gbz <= 1; gbz++){

	      boundaries_get_ghostbox(gbx,gby,gbz,&shiftx,&shifty,&shiftz,&shiftvx,&shiftvy,&shiftvz, OMEGA_dev,boxsize_dev, boxsize_dev, boxsize_dev, t_dev);
	 
	      depth = j;
	
	      node_local[j] = *num_nodes_dev;
	      pos_local[j] = 0;

	      for (cl_int wavefront_id = 0; wavefront_id < WAVEFRONT_SIZE; wavefront_id++){
		i = sort_dev[k + wavefront_id];	      
		collisions_dev[i + gbx_offset*(gbx+1) + gby_offset*(gby+1) + gbz_offset*(gbz+1)] = -1; 	}

	      while (depth >= j){
		while(pos_local[depth] < 8){
		  node = children_dev[node_local[depth]*8 + pos_local[depth]];
		  pos_local[depth]++;
		  children_local[base] = node;

		  if(node >= 0){
		    if (node < *num_bodies_dev){
		      nodex_local[base] = x_dev[node];
		      nodey_local[base] = y_dev[node];
		      nodez_local[base] = z_dev[node];
		      nodevx_local[base] = vx_dev[node];
		      nodevy_local[base] = vy_dev[node];
		      nodevz_local[base] = vz_dev[node];
		      noderad_local[base] = rad_dev[node];
		      nodem_local[base] = mass_dev[node];
		    }
		    
		    else{
		      nodex_local[base] = cellcenter_x_dev[node];
		      nodey_local[base] = cellcenter_y_dev[node];
		      nodez_local[base] = cellcenter_z_dev[node];
		    }
		  }

		  node = children_local[base];

		  if (node >= 0){

		    for (cl_int wavefront_id = 0; wavefront_id < WAVEFRONT_SIZE; wavefront_id++){
		      dx[wavefront_id] = (body_x[wavefront_id] + shiftx) - nodex_local[base];
		      dy[wavefront_id] = (body_y[wavefront_id] + shifty) - nodey_local[base];
		      dz[wavefront_id] = (body_z[wavefront_id] + shiftz) - nodez_local[base];
		      temp_register[wavefront_id] = dx[wavefront_id]*dx[wavefront_id];
		      temp_register[wavefront_id] += dy[wavefront_id]*dy[wavefront_id];
		      temp_register[wavefront_id] += dz[wavefront_id]*dz[wavefront_id];

		    }

		    if (node < *num_bodies_dev){
		      for (cl_int wavefront_id = 0; wavefront_id < WAVEFRONT_SIZE; wavefront_id++){
			i = sort_dev[k + wavefront_id];	 
			if (node != i){
			  float rp = body_rad[wavefront_id] + noderad_local[base];
			  dvx[wavefront_id] = (body_vx[wavefront_id] + shiftvx) - nodevx_local[base]; 
			  dvy[wavefront_id] = (body_vy[wavefront_id] + shiftvy) - nodevy_local[base];
			  dvz[wavefront_id] = (body_vz[wavefront_id] + shiftvz) - nodevz_local[base];        	
			  if ( temp_register[wavefront_id] <= rp*rp && dvx[wavefront_id]*dx[wavefront_id] + dvy[wavefront_id]*dy[wavefront_id] + dvz[wavefront_id]*dz[wavefront_id] < 0){	  
			    collisions_dev[i + gbx_offset*(gbx+1) + gby_offset*(gby+1) + gbz_offset*(gbz+1)] = node;
			  }	
			}     
		      }
		    }

		    
		    else{
		      for (cl_int wavefront_id = 0; wavefront_id < WAVEFRONT_SIZE; wavefront_id++){
			float rp = body_rad[wavefront_id] + dr_cutoff_local[depth];
			wavefront_vote_local[wavefront_leader_id + wavefront_id] = (temp_register[wavefront_id] >= rp*rp) ? 1 : 0;
		      }
		      for(l = 1; l < WAVEFRONT_SIZE; l++)
			wavefront_vote_local[wavefront_leader_id] += wavefront_vote_local[wavefront_leader_id+l];
		      if (wavefront_vote_local[wavefront_leader_id] < WAVEFRONT_SIZE){
			depth++;
			node_local[depth] = node;
			pos_local[depth] = 0;
		      }

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
	    }
	  }
	}
      }
    }
  }
}
