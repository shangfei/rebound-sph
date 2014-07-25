PROJ=tree_cl_test 
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
endif

INC_DIRS+=./
INC_DIRS+=../../src/
INC_PARAMS=$(INC_DIRS:%=-I%)
LIB_PARAMS=$(LIB_DIRS:%=-L%)

.PHONY: all clean

all: $(PROJ)

$(PROJ): tree_cl_test_main.o tree_cl_test.o cl_host_tools.o
	$(CC) -o $@ $^ $(INC_PARAMS) $(LIB_PARAMS) $(LIB)
tree_cl_test_main.o:
	$(CC) $(CFLAGS) -c -o $@ tree_cl_test_main.c $(INC_PARAMS) 
tree_cl_test.o: 
	$(CC) $(CFLAGS) -c -o $@ tree_cl_test.c $(INC_PARAMS) 
cl_host_tools.o:
	$(CC) $(CFLAGS) -c -o $@ ../../src/cl_host_tools.c $(INC_PARAMS) 
clean:
	rm -f $(PROJ) 
	rm -f *.o
	rm -f *~	
	rm -f *.*~
