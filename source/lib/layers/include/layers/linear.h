#ifndef QCONV_LAYERS_LINEAR_H_INCLUDED
#define QCONV_LAYERS_LINEAR_H_INCLUDED

#include <cassert>
#include <cstring>

#include <simd/simd.h>

static inline void m256_add_dpbusd_epi32x2(__m256i& acc, __m256i a0, __m256i b0, __m256i a1, __m256i b1)
{
  __m256i product0 = _mm256_maddubs_epi16(a0, b0);
  __m256i product1 = _mm256_maddubs_epi16(a1, b1);
  product0 = _mm256_madd_epi16(product0, _mm256_set1_epi16(1));
  product1 = _mm256_madd_epi16(product1, _mm256_set1_epi16(1));
  acc = _mm256_add_epi32(acc, _mm256_add_epi32(product0, product1));
}

namespace qconv::layers
{
template <size_t InSize, size_t OutSize>
class Linear
{
public:
  using InputType = uint8_t;
  using WeightType = int8_t;
  using OutputType = int32_t;

  void init(WeightType w[OutSize][InSize], OutputType b[OutSize])
  {
    for (size_t i = 0; i < OutSize; ++i) {
      for (size_t j = 0; j < InSize; ++j) {
        weights[j / 4][i * 4 + (j % 4)] = w[i][j];
      }
    }
    for (size_t i = 0; i < OutSize; ++i) {
      biases[i] = b[i];
    }
  }

  // Forward propagation
  void propagate(InputType* input)
  {
    assert(simd::isPtrAligned(input));
    static_assert(InSize % 4 == 0, "InSize must be divisble by 4");
    static_assert(OutSize % 8 == 0, "OutSize must be divisble by 8");
    constexpr size_t InStep = InSize / 4;
    constexpr size_t OutStep = OutSize / 8;

    const auto input32 = reinterpret_cast<const std::int32_t*>(input);
    const __m256i* biasesVec = reinterpret_cast<const __m256i*>(biases);
    __m256i acc[OutStep];
    for (size_t k = 0; k < OutStep; ++k) {
      acc[k] = biasesVec[k];
    }
    for (size_t i = 0; i < InStep; i += 2) {
      const __m256i in0 = _mm256_set1_epi32(input32[i + 0]);
      const __m256i in1 = _mm256_set1_epi32(input32[i + 1]);
      const auto row0 = reinterpret_cast<const __m256i*>(&weights[i][0]);
      const auto row1 = reinterpret_cast<const __m256i*>(&weights[i + 1][0]);
      for (size_t k = 0; k < OutStep; ++k) {
        m256_add_dpbusd_epi32x2(acc[k], in0, row0[k], in1, row1[k]);
      }
    }
    __m256i* outptr = reinterpret_cast<__m256i*>(outputBuf);
    for (size_t k = 0; k < OutStep; ++k) {
      outptr[k] = acc[k];
    }
  }

  alignas(simd::Alignment) WeightType weights[InSize / 4][OutSize * 4];
  alignas(simd::Alignment) OutputType biases[OutSize];
  alignas(simd::Alignment) OutputType outputBuf[OutSize];
};
}  // namespace qconv::layers

#endif  // #ifndef QCONV_LAYERS_LINEAR_H_INCLUDED