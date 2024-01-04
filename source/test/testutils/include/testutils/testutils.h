#ifndef QCONV_TESTUTILS_TESTUTILS_H_
#define QCONV_TESTUTILS_TESTUTILS_H_

#include <iostream>
#include <random>

#include <simd/simd.h>

namespace qconv::testutils
{
template <typename T>
void constInit(T* a, int s, T v)
{
  for (int i = 0; i < s; ++i) {
    a[i] = v;
  }
}

template <typename T>
void modInit(T* a, int s, int mod)
{
  for (int i = 0; i < s; ++i) {
    a[i] = i % mod;
  }
}

template <typename T>
void randInit(T* a, int s)
{
  std::mt19937 e;
  unsigned long long mod = (1UL << (sizeof(T) * 8));
  for (int i = 0; i < s; ++i) {
    int r = e() % mod;
    if (std::is_signed<T>::value) {
      r -= (mod / 2);
    }
    a[i] = r;
  }
}

template <typename T>
void weightInit_32_512(T a[32][512])
{
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 512; ++j) {
      a[i][j] = (i + j) % 128;
    }
  }
}

template <typename PrintType>
void printLongRegister(__m128i v)
{
  constexpr uint32_t N = sizeof(__m128i) / sizeof(PrintType);
  PrintType values[N];
  _mm_storeu_si128(values, v);
  for (int i = 0; i < N; ++i) {
    std::cout << static_cast<int>(values[i]) << " ";
  }
  std::cout << "\n";
}

template <typename PrintType>
void printLongRegister(__m256i v)
{
  constexpr uint32_t N = sizeof(__m256i) / sizeof(PrintType);
  PrintType values[N];
  _mm256_storeu_si256(values, v);
  for (int i = 0; i < N; ++i) {
    std::cout << static_cast<int>(values[i]) << " ";
  }
  std::cout << "\n";
}
}  // namespace qconv::testutils

#endif  // QCONV_TESTUTILS_TESTUTILS_H_
