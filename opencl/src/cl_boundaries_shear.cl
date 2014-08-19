
void cl_boundaries_get_ghostbox(
			     int i, 
			     int j, 
			     int k,
			     float* shiftx,
			     float* shifty,
			     float* shiftz,
			     float* shiftvx,
			     float* shiftvy,
			     float* shiftvz,
			     float OMEGA,
			     float boxsize_x,
			     float boxsize_y,
			     float boxsize_z,
			     float t
			     ){
  *shiftvx = 0.f;
  *shiftvy = -1.5f*(float)i*OMEGA*boxsize_x;
  *shiftvz = 0.f;
  float shift = (i == 0) ? 0.f : -fmod(*shiftvy*t - ((i>0) - (i<0))*boxsize_y/2.f, boxsize_y) -  ((i>0) - (i<0))*boxsize_y/2.f;
  *shiftx = boxsize_x*(float)i;
  *shifty = boxsize_y*(float)j-shift;
  *shiftz = boxsize_z*(float)k;
}

__kernel void cl_boundaries_check(
			       __global float* x_dev,
			       __global float* y_dev,
			       __global float* z_dev,
			       __global float* vx_dev,
			       __global float* vy_dev,
			       __global float* vz_dev,
			       __global float* t_dev,
			       __constant float* boxsize_dev,
			       __constant float* OMEGA,
			       __constant int* num_bodies_dev
			       ){
  
  int id,k, inc;
  __local float t_local;

  id = get_local_id(0);
  if (id == 0){
    t_local = *t_dev;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float t = t_local;
   
  id += get_group_id(0)*get_local_size(0);
  inc = get_global_size(0);

  
  float boxsize_x = *boxsize_dev;
  float boxsize_y = *boxsize_dev;
  float boxsize_z = *boxsize_dev;
  
  //offset of origin to touching six blocks to the right of main six blocks
  float offsetp1 = -fmod(-1.5f*(*OMEGA)*boxsize_x*t+boxsize_y/2.f,boxsize_y)-boxsize_y/2.f;
  
  //offset of origin to touching six blocks to the left of main six blocks
  float offsetm1 = -fmod( 1.5f*(*OMEGA)*boxsize_x*t-boxsize_y/2.f,boxsize_y)+boxsize_y/2.f;
   
  for (k = id; k < *num_bodies_dev; k += inc){
  
    // Radial
    while(x_dev[k] > boxsize_x/2.f){
      x_dev[k] -= boxsize_x;
      y_dev[k] += offsetp1;
      vy_dev[k] += 3.f/2.f*(*OMEGA)*boxsize_x;
    }
   
    while(x_dev[k] < -boxsize_x/2.f){
      x_dev[k] += boxsize_x;
      y_dev[k] += offsetm1;
      vy_dev[k] -= 3.f/2.f*(*OMEGA)*boxsize_x;
    }

    // Azimuthal
    while(y_dev[k] > boxsize_y/2.f){
      y_dev[k] -= boxsize_y;
    }
    while(y_dev[k] < -boxsize_y/2.f){
      y_dev[k] += boxsize_y;
    }
  
    // Vertical
    while(z_dev[k] > boxsize_z/2.f){
      z_dev[k] -= boxsize_z;
    }
    while(z_dev[k] < -boxsize_z/2.f){
      z_dev[k] += boxsize_z;
    }
  }

}
