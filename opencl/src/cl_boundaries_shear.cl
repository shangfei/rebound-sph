
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

/* __kernel void boundaries_check( */
/* 			       __global float* x_dev, */
/* 			       __global float* y_dev, */
/* 			       __global float* z_dev, */
/* 			       __global float* t, */
/* 			       __global float* boxsize_x_dev, */
/* 			       __global float* boxsize_y_dev, */
/* 			       __global float* boxsize_z_dev */
/* 			       ){ */
  
/*   unsigned int i = get_global_id(0); */
/*   float t = *t_dev; */

/*   //offset of origin to touching six blocks to the right of main six blocks */
/*   float offsetp1 = -fmod(-1.5*OMEGA*boxsize_x*t+boxsize_y/2.,boxsize_y)-boxsize_y/2.;  */

/*   //offset of origin to touching six blocks to the left of main six blocks */
/*   float offsetm1 = -fmod( 1.5*OMEGA*boxsize_x*t-boxsize_y/2.,boxsize_y)+boxsize_y/2.;  */
  
/*   // Radial */
/*   while(x_dev[i] > boxsize_x_dev/2.){ */
/*     x_dev[i] -= boxsize_x_dev; */
/*     y_dev[i] += offsetp1; */
/*     vy_dev[i] += 3./2.*OMEGA*boxsize_x_dev; */
/*   } */
  
/*   while(x_dev[i] < -boxsize_x_dev/2.){ */
/*     x_dev[i] += boxsize_x_dev; */
/*     y_dev[i] += offsetm1; */
/*     vy_dev[i] -= 3./2.*OMEGA*boxsize_x_dev; */
/*   } */
  
/*   // Azimuthal */
/*   while(y_dev[i] > boxsize_y_dev/2.){ */
/*     y_dev[i] -= boxsize_y_dev; */
/*   } */
/*   while(y_dev[i] < -boxsize_y_dev/2.){ */
/*     y_dev[i] += boxsize_y_dev; */
/*   } */
  
/*   // Vertical  */
/*   while(z_dev[i] > boxsize_z_dev/2.){ */
/*     z_dev[i] -= boxsize_z_dev; */
/*   } */
/*   while(z_dev[i] < -boxsize_z_dev/2.){ */
/*     z_dev[i] += boxsize_z_dev; */
/*   } */
  
/* } */



