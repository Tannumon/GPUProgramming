NVCC ?= nvcc
CXXSTD ?= c++17
OPTFLAGS ?= -O2

all: mm.exe

mm.exe: matmul.cu
	$(NVCC) $(OPTFLAGS) -o $@ $<

clean:
	rm -f mm.exe