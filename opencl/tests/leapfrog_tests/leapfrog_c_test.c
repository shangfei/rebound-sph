void cl_leapfrog_integrator_part1(float* x,
				  float* y,
				  float* z,
				  float* vx,
				  float* vy,
				  float* vz,
				  float dt
				  ){
  for(int i = 0; i < N; i++)
    x[i] += v[i]*0.5*dt;
}

void cl_leapfrog_integrator_part2(float* x, 
				  float* y,
				  float* z,
				  float* vx,
				  float* vy,
				  float* vz,
				  float dt
				  ){
  for(int i = 0; i < N; i++){
    v[i] += dt * x[i];
    x[i] += v[i]*0.5*dt;
  }
}


void leapfrog_c_test()
{
  


  return 0;
}
