#include <chrono>
#include <cstdlib>
#include <iostream>

// clang-format off
#ifndef N
#  define N 20
#endif

#ifndef ITRS
#  define ITRS 1
#endif
// clang-format on

__global__ void saxpy_cuda(size_t n, float a, float *x, float *y, float *z) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < n)
    z[idx] = (a * x[idx]) + y[idx];
}

int main(int argc, const char *argv[]) {
  using namespace std::chrono;
  size_t n;
  int itrs;
  if (argc < 3) {
    n = 1 << N;
    itrs = ITRS;
  } else {
    n = 1 << std::atoi(argv[1]);
    itrs = std::atoi(argv[2]);
  }

  // we need to use posix_memalign API here
  // since operator new does not guaratee alignment
  float a = 8.0f;
  float *x = static_cast<float *>(aligned_alloc(64, n * sizeof(float)));
  float *y = static_cast<float *>(aligned_alloc(64, n * sizeof(float)));
  float *z = static_cast<float *>(aligned_alloc(64, n * sizeof(float)));
  for (size_t i = 0; i < n; i++) {
    x[i] = 2.0f;
    y[i] = 2.0f;
    z[i] = 0.0f;
  }

  float *d_x;
  float *d_y;
  float *d_z;
  cudaMalloc(&d_x, n * sizeof(float));
  cudaMalloc(&d_y, n * sizeof(float));
  cudaMalloc(&d_z, n * sizeof(float));
  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, z, n * sizeof(float), cudaMemcpyHostToDevice);

  auto start = high_resolution_clock::now();
  for (auto i = 0; i < itrs; i++) {
    saxpy_cuda<<<(n + 255) / 256, 256>>>(n, a, d_x, d_y, d_z);
  }
  auto end = high_resolution_clock::now();

  auto diff = duration_cast<microseconds>(end - start);
  std::cout << "CUDA vectorized saxpy:\n"
            << "\tIterations = " << itrs << "\n\t"
            << "Average time = " << diff.count() / itrs
            << " micros. Total time= " << diff.count() << " micros.\n";
  return 0;
}