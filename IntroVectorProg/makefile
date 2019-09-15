CC  := gcc
CXX := g++
NVCC := nvcc

N := 24
ITRS := 5

default: all
all: scalar omp avx avx512 cuda
clean:
	rm -rf *.o *.out

scalar: saxpy.cpp
	$(CXX) saxpy.cpp -std=c++11 -march=native -DN=$(N) -O0 -DITRS=$(ITRS) -Wall -Wextra -o saxpy_scalar.out

omp: saxpy_omp.cpp
	$(CXX) saxpy_omp.cpp -std=c++11 -O1 -march=native -fopenmp -DN=$(N) -DITRS=$(ITRS) -Wall -Wextra -o saxpy_omp.out

avx: saxpy_avx.cpp
	$(CXX) saxpy_avx.cpp -std=c++11 -O1 -mavx2 -DN=$(N) -DITRS=$(ITRS) -Wall -Wextra -o saxpy_avx256.out

avx512: saxpy_avx.cpp
	$(CXX) saxpy_avx.cpp -std=c++11 -O1 -mavx512f -DN=$(N) -DITRS=$(ITRS) -Wall -Wextra -o saxpy_avx512.out

cuda: saxpy_cuda.cu
	$(NVCC) saxpy_cuda.cu -std=c++11 -DN=$(N) -DITRS=$(ITRS) -o saxpy_cuda.out

runall: all
	./saxpy_scalar.out
	./saxpy_omp.out
	./saxpy_avx256.out
	./saxpy_avx512.out
	./saxpy_cuda.out
