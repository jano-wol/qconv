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

T* copy(T* output, const T* input)
{
  typedef VecBatch B;
  typedef VecLoadStore LS;
  for (int i = 0; i < B::NumBatch; i++) {
    auto data = LS::load(input + i * B::RegWidth);
    LS::store(output + i * B::RegWidth, data);
  }

  return output + B::NumBatch * B::RegWidth;
}

int main()
{
  alignas(32) T a[Size];
  alignas(32) T b[Size];

  for (int i = 0; i < Size; ++i) {
    std::cin >> a[i];
  }

  copy(b, a);
  for (int i = 0; i < Size; ++i) {
    std::cout << b[i];
  }
  std::cout << "\n";
}
