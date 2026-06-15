# Makefile — build vectoradd
NVCC ?= nvcc
ARCH ?= native
CXXSTD ?= c++17
OPTFLAGS ?= -O2
LIBS      := -lcublas

all: vectoradd

vectoradd: vectoradd.cu
	$(NVCC) $(OPTFLAGS) -std=$(CXXSTD) -arch=$(ARCH) -o $@ $< $(LIBS)

clean:
	rm -f vectoradd


