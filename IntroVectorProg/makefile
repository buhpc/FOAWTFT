N := 24
ITRS := 5

default: all
all: scalar omp avx cuda
clean:
	rm -rf *.o *.out

scalar: saxpy.cpp
	$(CXX) saxpy.cpp -std=c++11 -march=native -DN=$(N) -fno-exceptions \
		-fno-rtti -O0 -DITRS=$(ITRS) -Wall -Wextra -o saxpy_scalar.out

omp: saxpy_omp.cpp
	$(CXX) saxpy_omp.cpp -std=c++11 -O1 -march=native  -fno-exceptions \
		-fno-rtti -fopenmp -DN=$(N) -DITRS=$(ITRS) -Wall -Wextra \
		-o saxpy_omp.out

avx: saxpy_avx.cpp
	$(CXX) saxpy_avx.cpp -g -std=c++11 -O1 \
		-march=native -DN=$(N) -DITRS=$(ITRS) -Wall -Wextra \
		-fno-exceptions -fno-rtti -o saxpy_avx.out

cuda: saxpy_cuda.cu
	$(NVCC) saxpy_cuda.cu -std=c++11 -DN=$(N) -DITRS=$(ITRS) -o saxpy_cuda.out

runall: all
	./saxpy_scalar.out
	./saxpy_omp.out
	./saxpy_avx.out
	./saxpy_cuda.out
