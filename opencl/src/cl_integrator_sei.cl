void operator_H012(
		   int i,
		   float dt,
		   float OMEGAZ,
		   float OMEGA,
		   float sindt,
		   float tandt,
		   float sindtz,
		   float tandtz,
		   __global float* x_dev,
		   __global float* y_dev,
		   __global float* z_dev,
		   __global float* vx_dev,
		   __global float* vy_dev,
		   __global float* vz_dev,
		   __global float* ax_dev,
		   __global float* ay_dev,
		   __global float* az_dev
		   )
{
  /* Integrate vertical  motion */
  float zx = z_dev[i] * OMEGAZ;
  float zy = vz_dev[i];

  /* Rotation implemented as 3 shear operators to avoid roundoff errors */
  float zt1 = zx - tandtz*zy;
  float zyt = sindtz*zt1 + zy;
  float zxt = zt1 - tandtz*zyt;
  z_dev[i] = zxt/OMEGAZ;
  vz_dev[i] = zyt;

  float a0 = 2.f*vy_dev[i] + 4.f*x_dev[i]*OMEGA;
  float b0 = y_dev[i]*OMEGA -2.f*vx_dev[i];

  float ys = (y_dev[i]*OMEGA-b0)/2.f;
  float xs = (x_dev[i]*OMEGA-a0);

  float xst1 = xs - tandt*ys;
  float yst = sindt*xst1 + ys;
  float xst = xst1 - tandt*yst;

  x_dev[i] = (xst + a0) / OMEGA;
  y_dev[i] = (yst*2.f + b0) / OMEGA - 3.f/4.f*a0*dt;
  vx_dev[i] = yst;
  vy_dev[i] = -xst*2.f - 3.f/2.f*a0;
}

void operator_phi1(
		   int i,
		   float dt,
		   __global float* vx_dev,
		   __global float* vy_dev,
		   __global float* vz_dev,
		   __global float* ax_dev,
		   __global float* ay_dev,
		   __global float* az_dev
		   )
{
  //kick step
  vx_dev[i] += ax_dev[i] * dt;
  vy_dev[i] += ay_dev[i] * dt;
  vz_dev[i] += az_dev[i] * dt;
}

__kernel void cl_integrator_part1(
				    __global float* x_dev,
				    __global float* y_dev,
				    __global float* z_dev,
				    __global float* vx_dev,
				    __global float* vy_dev,
				    __global float* vz_dev,
				    __global float* ax_dev,
				    __global float* ay_dev,
				    __global float* az_dev,
				    __global float* t_dev,
				    __constant int* num_bodies_dev,
				    __constant float* dt_dev,
				    __constant float* OMEGA_dev,
				    __constant float* OMEGAZ_dev,
				    __constant float* sindt_dev,
				    __constant float* tandt_dev,
				    __constant float* sindtz_dev,
				    __constant float* tandtz_dev
				  )
{
  __local float t_local;
  int id,k, inc;

  id = get_local_id(0);
  if (id == 0){
    t_local = *t_dev;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float t = t_local;

  id += get_group_id(0)*get_local_size(0);
  inc = get_global_size(0);

  for (k = id; k < *num_bodies_dev; k += inc){
    operator_H012(
		  k, 
		  *dt_dev,
		  *OMEGA_dev,
		  *OMEGAZ_dev,
		  *sindt_dev,
		  *tandt_dev,
		  *sindtz_dev,
		  *tandtz_dev,
		  x_dev, 
		  y_dev, 
		  z_dev, 
		  vx_dev, 
		  vy_dev, 
		  vz_dev, 
		  ax_dev, 
		  ay_dev, 
		  az_dev
		  );
  }

  t += *dt_dev/2.f;
  if (id == 0) 
    *t_dev = t;
}

__kernel void cl_integrator_part2(
				    __global float* x_dev,
				    __global float* y_dev,
				    __global float* z_dev,
				    __global float* vx_dev,
				    __global float* vy_dev,
				    __global float* vz_dev,
				    __global float* ax_dev,
				    __global float* ay_dev,
				    __global float* az_dev,
				    __global float* t_dev,
				    __constant int* num_bodies_dev,
				    __constant float* dt_dev,
				    __constant float* OMEGA_dev,
				    __constant float* OMEGAZ_dev,
				    __constant float* sindt_dev,
				    __constant float* tandt_dev,
				    __constant float* sindtz_dev,
				    __constant float* tandtz_dev
				  )
{
  __local float t_local;
  int id,k, inc;

  id = get_local_id(0);
  if (id == 0){
    t_local = *t_dev;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float t = t_local;

  id += get_group_id(0)*get_local_size(0);
  inc = get_global_size(0);

  for (k = id; k < *num_bodies_dev; k += inc){
    operator_phi1(k, *dt_dev, vx_dev, vy_dev, vz_dev, ax_dev, ay_dev, az_dev);
    operator_H012(k, 
		  *dt_dev,
		  *OMEGA_dev,
		  *OMEGAZ_dev,
		  *sindt_dev,
		  *tandt_dev,
		  *sindtz_dev,
		  *tandtz_dev,
		  x_dev, 
		  y_dev, 
		  z_dev, 
		  vx_dev, 
		  vy_dev, 
		  vz_dev, 
		  ax_dev, 
		  ay_dev, 
		  az_dev
		  );
  }

  t += *dt_dev/2.f;
  if (id == 0) 
    *t_dev = t;
}

