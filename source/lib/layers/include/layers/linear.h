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
  using InputType = int8_t;
  using WeightType = int8_t;
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
    assert(simd::isPtrAligned(outputBuf));
    assert(simd::isPtrAligned(input));
    assert(simd::isPtrAligned(weights));
    assert(simd::isPtrAligned(biases));
    std::memcpy(outputBuf, biases, OutSize * sizeof(OutputType));
    static_assert(InSize % 8 == 0, "InSize must be divisble by 8");
    for (size_t inIdx = 0; inIdx < InSize; inIdx += 8) {
      auto inputChunk = _mm256_setr_epi32(input[inIdx], input[inIdx + 1], input[inIdx + 2], input[inIdx + 3],
                                          input[inIdx + 4], input[inIdx + 5], input[inIdx + 6], input[inIdx + 7]);
      for (size_t outIdx = 0; outIdx < OutSize; ++outIdx) {
        auto weightsChunk =
            _mm256_setr_epi32(weights[outIdx][inIdx], weights[outIdx][inIdx + 1], weights[outIdx][inIdx + 2],
                              weights[outIdx][inIdx + 3], weights[outIdx][inIdx + 4], weights[outIdx][inIdx + 5],
                              weights[outIdx][inIdx + 6], weights[outIdx][inIdx + 7]);
        auto x = _mm256_mullo_epi32(inputChunk, weightsChunk);
        auto y = hsum_8x32(x);
        outputBuf[outIdx] += y;
      }
    }
  }

  alignas(simd::Alignment) WeightType weights[OutSize][InSize];
  alignas(simd::Alignment) OutputType biases[OutSize];
  alignas(simd::Alignment) OutputType outputBuf[OutSize];
};
}  // namespace qconv::layers

#endif  // #ifndef QCONV_LAYERS_LINEAR_H_INCLUDED