__kernel void cl_leapfrog_integrator_part1(__global float* x,
					   __global float* y,
					   __global float* z,
					   __global float* vx,
					   __global float* vy,
					   __global float* vz,
					   float dt
					   ){
  unsigned int i = get_global_id(0);
  x[i] += v[i]*0.5*dt;
  //t += dt/2;
}

__kernel void cl_leapfrog_integrator_part2(__global float* x, 
					   __global float* y,
					   __global float* z,
					   __global float* vx,
					   __global float* vy,
					   __global float* vz,
					   float dt
					   ){
  unsigned int i = get_global_id(0);
  v[i] += dt * x[i];
  x[i] += v[i]*0.5*dt;
  //t += dt/2;
}







