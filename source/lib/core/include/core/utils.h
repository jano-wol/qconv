#ifndef QCONV_CORE_UTILS_H_
#define QCONV_CORE_UTILS_H_

namespace qconv::core
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
void weightInit_32_512(T a[32][512])
{
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 512; ++j) {
      a[i][j] = (i + j) % 128;
    }
  }
}
}  // namespace qconv::core

#endif  // QCONV_CORE_UTILS_H_
