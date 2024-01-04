// x86-64 gcc 11.4 -O3 -mavx2
#include <immintrin.h>
#include <iostream>

constexpr uint32_t Size = 4096;
using T = int32_t;

struct VecBatch
{
  static constexpr uint32_t SimdBits = 256;
  static constexpr uint32_t RegWidth = SimdBits ? (SimdBits / 8) / sizeof(T) : 1;
  static constexpr uint32_t NumBatch = Size / RegWidth;
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
  static inline __attribute__((always_inline))  R min(R a, R b) { return _mm256_min_epi32(a, b); }
};

__m256i minH(T* a)
{
  typedef VecBatch B;
  typedef VecLoadStore LS;
  typedef VecOp Op;

  auto currMin = LS::load(a);
  for (int i = 1; i < B::NumBatch; i++) {
    auto next = LS::load(a + i * B::RegWidth);
    currMin = Op::min(currMin, next);
  }
  return currMin;
}

int main()
{
  using T = int32_t;

  alignas(32) T a[Size];

  for (int i = 0; i < Size; ++i) {
    std::cin >> a[i];
  }

  auto ret = minH(a);
  std::cout << _mm256_extract_epi16(ret, 1) << "\n";
}
