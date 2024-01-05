#ifndef QCONV_LAYERS_LINEAR_H_INCLUDED
#define QCONV_LAYERS_LINEAR_H_INCLUDED

#include <cassert>
#include <cstring>

#include <simd/simd.h>

static inline int32_t hsum_epi32(__m128i x)
{
  __m128i hi64 = _mm_unpackhi_epi64(x, x);
  __m128i sum64 = _mm_add_epi32(hi64, x);
  __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));  // Swap the low two elements
  __m128i sum32 = _mm_add_epi32(sum64, hi32);
  return _mm_cvtsi128_si32(sum32);  // movd
}

static inline int32_t hsum_8x32(__m256i v)
{
  __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  return hsum_epi32(sum128);
}

namespace qconv::layers
{
template <size_t InSize, size_t OutSize>
class Linear
{
public:
  using InputType = int32_t;
  using WeightType = int32_t;
  using OutputType = int32_t;

  void init(WeightType w[OutSize][InSize], OutputType b[OutSize])
  {
    for (size_t i = 0; i < OutSize; ++i) {
      for (size_t j = 0; j < InSize; ++j) {
        weights[i][j] = w[i][j];
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
    std::memcpy(outputBuf, biases, OutSize * sizeof(OutputType));
    static_assert(InSize % 8 == 0, "InSize must be divisble by 8");
    for (size_t outIdx = 0; outIdx < OutSize; ++outIdx) {
      auto sum = _mm256_setzero_si256();
      for (size_t inIdx = 0; inIdx < InSize; inIdx += 8) {
        auto inputChunk = _mm256_load_si256(reinterpret_cast<__m256i const*>(&input[inIdx]));
        auto weightsChunk = _mm256_load_si256(reinterpret_cast<__m256i const*>(&weights[outIdx][inIdx]));
        auto x = _mm256_mullo_epi32(inputChunk, weightsChunk);
        sum = _mm256_add_epi32(sum, x);
      }
      outputBuf[outIdx] += hsum_8x32(sum);
    }
  }

  alignas(simd::Alignment) WeightType weights[OutSize][InSize];
  alignas(simd::Alignment) OutputType biases[OutSize];
  alignas(simd::Alignment) OutputType outputBuf[OutSize];
};
}  // namespace qconv::layers

#endif  // #ifndef QCONV_LAYERS_LINEAR_H_INCLUDED