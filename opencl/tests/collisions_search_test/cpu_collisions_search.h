#ifndef CPU_COLLISION_SEARCH_H
#define CPU_COLLISION_SEARCH_H

void boundaries_get_ghostbox(
			     int , 
			     int , 
			     int ,
			     float *,
			     float *,
			     float *,
			     float *,
			     float *,
			     float *,
			     float *,
			     float *,
			     float *,
			     float *,
			     float *
			     );



void cpu_collisions_search(	
			   cl_float *, 
			   cl_float *,
			   cl_float *,
			   cl_float *,
			   cl_float *,
			   cl_float *,
			   cl_float *,
			   cl_int *,
			   cl_int *,
			   cl_int *,
			   cl_int *,
			   cl_int *,
			   cl_float*,
			   cl_float*,
			   cl_float*
				);




void cpu_collisions_search_2(	
			     cl_float *, 
			     cl_float *,
			     cl_float *,
			     cl_float*,
			     cl_float*,
			     cl_float*,
			     cl_float *,
			     cl_float *,
			     cl_float *,
			     cl_float *,
			     cl_float*,
			     cl_int* ,
			     cl_int*,
			     cl_int*,
			     cl_int*,
			     cl_float*,
			     cl_int*, 
			     cl_int*,
			     cl_int*, 
			     cl_int*,
			     cl_int*,
			     cl_float*,
			     cl_float*,
			     cl_float*,
			     cl_float*,
			     cl_float*,
			     cl_float*,
			     cl_float*,
			     cl_float*,
			     cl_float*,
			     cl_int* ,
			     cl_float*,
			     cl_float*,
			     cl_float*,
			     cl_int
				);


#endif
