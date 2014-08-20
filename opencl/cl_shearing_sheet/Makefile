UNAME := $(shell uname)
PROJ=cl_shearing_sheet
CFLAGS=-std=c99 -Wall
ifeq ($(UNAME), Darwin)
export LIB=-framework OpenCL
endif
ifeq ($(UNAME), Linux)
CC=gcc
LIB=-lOpenCL -lm
endif

ifdef DEBUG
	CFLAGS+=-g
endif

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
INC_DIRS+=../src/
INC_PARAMS=$(INC_DIRS:%=-I%)
LIB_PARAMS=$(LIB_DIRS:%=-L%)

.PHONY: all clean

all: $(PROJ)

$(PROJ): main.o cl_shearing_sheet.o cl_host_tools.o cl_init.o cl_tests.o
	$(CC) -o $@ $^ $(INC_PARAMS) $(LIB_PARAMS) $(LIB)
main.o: main.c
	$(CC) $(CFLAGS) -c -o $@ $^ $(INC_PARAMS) 
cl_shearing_sheet.o: cl_shearing_sheet.c
	$(CC) $(CFLAGS) -c -o $@ $^ $(INC_PARAMS) 
cl_init.o: ../src/cl_init.c
	$(CC) $(CFLAGS) -c -o $@ $^ $(INC_PARAMS)
cl_tests.o: ../src/cl_tests.c
	$(CC) $(CFLAGS) -c -o $@ $^ $(INC_PARAMS)
cl_host_tools.o: ../src/cl_host_tools.c
	$(CC) $(CFLAGS) -c -o $@ $^ $(INC_PARAMS) 
clean:	
	rm -f $(PROJ) 
	rm -f *.o
	rm -f *~	
	rm -f *.*~
