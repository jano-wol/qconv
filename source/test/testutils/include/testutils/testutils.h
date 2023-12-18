#ifndef QCONV_TESTUTILS_TESTUTILS_H_
#define QCONV_TESTUTILS_TESTUTILS_H_

#include <random>

#include <simdops/simdops.h>

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
  long long mod = (1 << (sizeof(T) * 8));
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

void printAs32s(simde__m256i v);
void printAs16s(simde__m256i v);
void printAs8s(simde__m256i v);
}  // namespace qconv::testutils

#endif  // QCONV_TESTUTILS_TESTUTILS_H_
