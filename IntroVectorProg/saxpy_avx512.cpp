#include <chrono>
#include <cstdlib>
#include <iostream>
#include <immintrin.h>

// clang-format off
// #if !defined(__AVX__)
// #  error "No AVX support detected"
// #else
#  if defined(__AVX512F__)  
#    define saxpy_avx saxpy_avx512
//#  elif defined(__AVX2__)
//#    define saxpy_avx saxpy_avx256
// #  else // __AVX__
// #    define saxpy_avx saxpy_avx128
// #  endif
#endif

#ifndef N
#  define N 20
#endif

#ifndef ITRS
#  define ITRS 1
#endif
// clang-format on

// #if defined(__AVX__)
// void saxpy_avx128(size_t n, float a, float *x, float *y, float *z) {
//   __m128 a_vec = _mm_set1_ps(a);

//   for (size_t i = 0; i < n; i += sizeof(__m128) / sizeof(float)) {
//     __m128 x_vec = _mm_load_ps(&x[i]);
//     __m128 y_vec = _mm_load_ps(&y[i]);
//     __m128 res_vec = _mm_add_ps(_mm_mul_ps(a_vec, x_vec), y_vec);
//     _mm_store_ps(&z[i], res_vec);
//   }
// }
// #endif

// #if defined(__AVX2__)
void saxpy_avx512(size_t n, float a, float *x, float *y, float *z) {
  __m512 a_vec = _mm512_set1_ps(a);

  for (size_t i = 0; i < n; i += sizeof(__m512) / sizeof(float)) {
    __m512 x_vec = _mm512_load_ps(&x[i]);
    __m512 y_vec = _mm512_load_ps(&y[i]);
    __m512 res_vec = _mm512_add_ps(_mm512_mul_ps(a_vec, x_vec), y_vec);
    _mm512_store_ps(&z[i], res_vec);
  }
}
// #endif

// #if defined(__AVX512F__)
// void saxpy_avx512(size_t n, float a, float *x, float *y, float *z) {
//   __m512 a_vec = _mm512_set1_ps(a);

//   for (size_t i = 0; i < n; i += sizeof(__m512) / sizeof(float)) {
//     __m512 x_vec = _mm512_load_ps(&x[i]);
//     __m512 y_vec = _mm512_load_ps(&y[i]);
//     __m512 res_vec = _mm512_add_ps(_mm512_mul_ps(a_vec, x_vec), y_vec);
//     _mm512_store_ps(&z[i], res_vec);
//   }
// }
// #endif

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
    saxpy_avx512(n, a, x, y, z);
  }
  auto end = high_resolution_clock::now();

  auto diff = duration_cast<milliseconds>(end - start);
  std::cout << "AVX vectorized saxpy:\n"
            << "\tIterations = " << itrs << "\n\t"
            << "Average time = " << diff.count() / itrs
            << " millis. Total time= " << diff.count() << " millis.\n";
  return 0;
}
