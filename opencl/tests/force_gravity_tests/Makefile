PROJ=force_gravity_cl_test 
CC=gcc
CFLAGS=-std=c99 -Wall
LIB=-lOpenCL

ifdef AMDAPPSDKROOT
	INC_DIRS=$(AMDAPPSDKROOT)/include
	LIB_DIRS=$(AMDAPPSDKROOT)/lib/x86
else

ifdef NVSDKCOMPUTE_ROOT
	INC_DIRS=$(NVSDKCOMPUTE_ROOT)/include
	LIB_DIRS=./
endif

ifdef SCINET_CUDA_INSTALL
        INC_DIRS=$(SCINET_CUDA_INSTALL)/include
        LIB_DIRS=./
endif

endif

INC_DIRS+=./
INC_DIRS+=../../src/
INC_PARAMS=$(INC_DIRS:%=-I%)
LIB_PARAMS=$(LIB_DIRS:%=-L%)

.PHONY: all clean

all: $(PROJ)

$(PROJ): force_gravity_cl_test_main.o force_gravity_cl_test.o cl_host_tools.o 
	$(CC) -o $@ $^ $(INC_PARAMS) $(LIB_PARAMS) $(LIB)
force_gravity_cl_test_main.o: force_gravity_cl_test_main.c
	$(CC) $(CFLAGS) -c -o $@ $^ $(INC_PARAMS) 
force_gravity_cl_test.o: force_gravity_cl_test.c
	$(CC) $(CFLAGS) -c -o $@ $^ $(INC_PARAMS) 
cl_host_tools.o: ../../src/cl_host_tools.c
	$(CC) $(CFLAGS) -c -o $@ $^ $(INC_PARAMS) 
clean:
	rm -f $(PROJ) 
	rm -f *.o
	rm -f *~	
	rm -f *.*~
