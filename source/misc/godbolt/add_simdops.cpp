// x86-64 gcc 11.4 -O3 -mavx2
#include <immintrin.h>
#include <iostream>

constexpr size_t Size = 4096;
using T = int32_t;

struct VecBatch
{
  static constexpr size_t SimdBits = 256;
  static constexpr size_t RegWidth = SimdBits ? (SimdBits / 8) / sizeof(T) : 1;
  static constexpr size_t NumBatch = Size / RegWidth;
};

struct VecLoadStore
{
  static inline __attribute__((always_inline)) auto load(const void* addr)
  {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(addr));
  }

  static inline __attribute__((always_inline)) void store(void* addr, __m256i data)
  {
    _mm256_store_si256(reinterpret_cast<__m256i*>(addr), data);
  }
};

struct VecOpSIAVX2
{
  typedef __m256i R;
  static inline __attribute__((always_inline)) R setzero() { return _mm256_setzero_si256(); }
  static inline __attribute__((always_inline)) R bitwiseor(R a, R b) { return _mm256_or_si256(a, b); }
  static inline __attribute__((always_inline)) R bitwiseand(R a, R b) { return _mm256_and_si256(a, b); }
  static inline __attribute__((always_inline)) R bitwisexor(R a, R b) { return _mm256_xor_si256(a, b); }
};

struct VecOp : VecOpSIAVX2
{
  typedef int32_t T;
  static inline __attribute__((always_inline))  R add(R a, R b) { return _mm256_add_epi32(a, b); }
};

T* add(T* output, const T* input0, const T* input1)
{
  typedef VecBatch B;
  typedef VecLoadStore LS;
  typedef VecOp Op;

  for (int i = 0; i < B::NumBatch; i++) {
    auto data0 = LS::load(input0 + i * B::RegWidth);
    auto data1 = LS::load(input1 + i * B::RegWidth);
    data0 = Op::add(data0, data1);
    LS::store(output + i * B::RegWidth, data0);
  }

  return output + B::NumBatch * B::RegWidth;
}

int main()
{
  alignas(32) T a[Size];
  alignas(32) T b[Size];
  alignas(32) T c[Size];

  for (int i = 0; i < Size; ++i) {
    std::cin >> a[i];
    std::cin >> b[i];
  }

  add(c, a, b);
  for (int i = 0; i < Size; ++i) {
    std::cout << c[i];
  }
  std::cout << "\n";
}
