#include <chrono>
#include <cstdlib>
#include <iostream>
#include <omp.h>
// clang-format off
#ifndef N
#  define N 20
#endif

#ifndef ITRS
#  define ITRS 1
#endif
// clang-format on

void saxpy_omp(size_t n, float a, float *__restrict__ x, float *__restrict__ y,
               float *__restrict__ z) {
//#pragma omp simd //aligned(x, y, z : 64) safelen(8)
#pragma omp simd
  for (size_t i = 0; i < n; i++) {
    z[i] = (a * x[i]) + y[i];
  }
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

  auto start = high_resolution_clock::now();
  for (auto i = 0; i < itrs; i++) {
    saxpy_omp(n, a, x, y, z);
  }
  auto end = high_resolution_clock::now();

  auto diff = duration_cast<milliseconds>(end - start);
  std::cout << "OMP vectorized saxpy:\n"
            << "\tNumber of Elements: " << (1 << N) << "\n"
            << "\tIterations = " << itrs << "\n\t"
            << "Average time = " << diff.count() / itrs
            << " millis. Total time= " << diff.count() << " millis.\n";
  return 0;
}
