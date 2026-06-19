NVCC ?= nvcc
CXXSTD ?= c++17
OPTFLAGS ?= -O2

all: hist.exe

hist.exe: histogram.cu
	$(NVCC) $(OPTFLAGS) -std=$(CXXSTD) -o $@ $<
clean:
	rm -f hist.exe