#ifndef QCONV_TESTUTILS_TESTUTILS_H_
#define QCONV_TESTUTILS_TESTUTILS_H_

#include <iostream>
#include <random>

#include <simd/simd.h>

namespace qconv::testutils
{
template <typename T>
void constInit(T* a, size_t s, T v)
{
  for (size_t i = 0; i < s; ++i) {
    a[i] = v;
  }
}

template <typename T, size_t X, size_t Y>
void constInit(T a[X][Y], T v)
{
  for (size_t i = 0; i < X; ++i) {
    for (size_t j = 0; j < Y; ++j) {
      a[i][j] = v;
    }
  }
}

template <typename T>
void modInit(T* a, size_t s, int mod)
{
  for (size_t i = 0; i < s; ++i) {
    a[i] = i % mod;
  }
}

template <typename T, size_t X, size_t Y>
void modInit(T a[X][Y], int mod)
{
  size_t n = 0;
  for (size_t i = 0; i < X; ++i) {
    for (size_t j = 0; j < Y; ++j) {
      a[i][j] = (i + j) % mod;
      ++n;
    }
  }
}

template <typename T>
void randInit(T* a, size_t s)
{
  std::mt19937 e;
  unsigned long long mod = (1UL << (sizeof(T) * 8));
  for (size_t i = 0; i < s; ++i) {
    int r = e() % mod;
    if (std::is_signed<T>::value) {
      r -= (mod / 2);
    }
    a[i] = r;
  }
}

template <typename T, int64_t LowerBound, int64_t UpperBound>
void randInit(T* a, size_t s)
{
  static_assert(UpperBound > LowerBound, "UpperBound must be at least LowerBound");
  std::mt19937 e;
  int64_t mod = UpperBound - LowerBound + 1;
  for (size_t i = 0; i < s; ++i) {
    int64_t r = e() % mod;
    a[i] = static_cast<T>(LowerBound + r);
  }
}

template <typename T, size_t X, size_t Y>
void randInit(T a[X][Y])
{
  std::mt19937 e;
  unsigned long long mod = (1UL << (sizeof(T) * 8));
  for (size_t i = 0; i < X; ++i) {
    for (size_t j = 0; j < Y; ++j) {
      int r = e() % mod;
      if (std::is_signed<T>::value) {
        r -= (mod / 2);
      }
      a[i][j] = r;
    }
  }
}

template <typename T, size_t X, size_t Y, int64_t LowerBound, int64_t UpperBound>
void randInit(T a[X][Y])
{
  static_assert(UpperBound > LowerBound, "UpperBound must be at least LowerBound");
  std::mt19937 e;
  int64_t mod = UpperBound - LowerBound + 1;
  for (size_t i = 0; i < X; ++i) {
    for (size_t j = 0; j < Y; ++j) {
      int r = e() % mod;
      a[i][j] = static_cast<T>(LowerBound + r);
    }
  }
}

template <typename PrintType>
void printLongRegister(__m128i v)
{
  constexpr size_t N = sizeof(__m128i) / sizeof(PrintType);
  PrintType values[N];
  _mm_storeu_si128(reinterpret_cast<__m128i_u*>(values), v);
  for (size_t i = 0; i < N; ++i) {
    std::cout << static_cast<int>(values[i]) << " ";
  }
  std::cout << "\n";
}

template <typename PrintType>
void printLongRegister(__m256i v)
{
  constexpr size_t N = sizeof(__m256i) / sizeof(PrintType);
  PrintType values[N];
  _mm256_storeu_si256(reinterpret_cast<__m256i_u*>(values), v);
  for (size_t i = 0; i < N; ++i) {
    std::cout << static_cast<int>(values[i]) << " ";
  }
  std::cout << "\n";
}
}  // namespace qconv::testutils

#endif  // QCONV_TESTUTILS_TESTUTILS_H_
