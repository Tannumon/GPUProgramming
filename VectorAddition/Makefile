NVCC ?= nvcc
CXXSTD ?= c++17
OPTFLAGS ?= -O2

all: vectoradd.exe

vectoradd.exe: vectoradd.cu
	$(NVCC) $(OPTFLAGS) -std=$(CXXSTD) -o vectoradd.exe vectoradd.cu

clean:
	del /Q vectoradd.exe