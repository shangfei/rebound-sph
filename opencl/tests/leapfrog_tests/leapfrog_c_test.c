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

bool leapfrog_c_test_compare_floats(float x1, float x2, eps){
  return fabs(x1 - x2) < eps;
}


bool leapfrog_c_test_check(float x[],
			   float y[],
			   float z[],
			   float vx[],
			   float vy[],
			   float vz[],
			   float x_dev[],
			   float y_dev[],
			   float z_dev[],
			   float vx_dev[],
			   float vy_dev[],
			   float vz_dev[]){
  
  float eps = .0000001;
  bool pass = 1;
  for (int i = 0; i < N; i++)
    if (leapfrog_c_test_compare_floats(x[i],x_dev[i], eps) == 0)
      pass = 0;
  return pass;
}


void leapfrog_c_test(float x [], float y [], float z [], float vx [], float vy [], float vz [], float dt)
{
  cl_leapfrog_integrator_part1(x,y,z,vx,vy,vz,dt);
  cl_leapfrog_integrator_part2(x,y,z,vx,vy,vz,dt);
  return 0;
}
