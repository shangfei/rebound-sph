__kernel void cl_integrator_leapfrog_part1(__global float* x,
					   __global float* y,
					   __global float* z,
					   __global float* vx,
					   __global float* vy,
					   __global float* vz,
					   float dt
					   ){
  unsigned int i = get_global_id(0);
  x[i] += vx[i]*0.5*dt;
  y[i] += vy[i]*0.5*dt;
  z[i] += vz[i]*0.5*dt;
  //t += dt/2;
}

__kernel void cl_integrator_leapfrog_part2(__global float* x, 
					   __global float* y,
					   __global float* z,
					   __global float* vx,
					   __global float* vy,
					   __global float* vz,
					   float dt
					   ){
  unsigned int i = get_global_id(0);
  vx[i] += dt * x[i];
  vy[i] += dt * y[i];
  vz[i] += dt * z[i];
  x[i] += vx[i]*0.5*dt;
  y[i] += vy[i]*0.5*dt;
  z[i] += vz[i]*0.5*dt;
  //t += dt/2;
}







