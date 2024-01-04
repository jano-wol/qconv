#ifndef QCONV_LAYERS_LINEAR_H_INCLUDED
#define QCONV_LAYERS_LINEAR_H_INCLUDED

#include <cassert>

#include <simd/simd.h>

static inline __m128i hsumI32x4(__m256i sum0, __m256i sum1, __m256i sum2, __m256i sum3)
{
  sum0 = _mm256_hadd_epi32(sum0, sum1);
  sum2 = _mm256_hadd_epi32(sum2, sum3);
  sum0 = _mm256_hadd_epi32(sum0, sum2);

  __m128i sum128lo = _mm256_castsi256_si128(sum0);
  __m128i sum128hi = _mm256_extracti128_si256(sum0, 1);
  __m128i sum128 = _mm_add_epi32(sum128lo, sum128hi);

  return sum128;
}

static inline void add_dpbusd_epi32(__m256i& acc, __m256i a, __m256i b)
{
  // #if defined(USE_VNNI)
  //   This does exactly the same thing as explained below but in one instruction.
  //   acc = _mm256_dpbusd_epi32(acc, a, b);
  __m256i product = _mm256_maddubs_epi16(a, b);
  product = _mm256_madd_epi16(product, _mm256_set1_epi16(1));
  acc = _mm256_add_epi32(acc, product);
};

namespace qconv::layers
{
template <size_t InSize, size_t OutSize>
class Linear
{
public:
  using InputType = int8_t;
  using WeightType = int8_t;
  using OutputType = int32_t;

  void init(WeightType* w, OutputType* b)
  {
    for (size_t i = 0; i < InSize * OutSize; ++i) {
      weights[i / InSize][i % InSize] = w[i];
    }
    for (size_t i = 0; i < OutSize; ++i) {
      biases[i] = b[i];
    }
  }

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
    static_assert(OutSize % 4 == 0, "OutSize must be divisble by 4");
    constexpr int OutNumBatches = OutSize / 4;
    for (int i = 0; i < OutNumBatches; i++) {
      // Prepare weight offsets. One offset for one row of weights.
      // This is a simple index into a 2d array.
      const int offset0 = (i * 4 + 0) * InSize;
      const int offset1 = (i * 4 + 1) * InSize;
      const int offset2 = (i * 4 + 2) * InSize;
      const int offset3 = (i * 4 + 3) * InSize;

      // Accumulation starts from 0, we add the bias only at the end.
      auto sum0 = _mm256_setzero_si256();
      auto sum1 = _mm256_setzero_si256();
      auto sum2 = _mm256_setzero_si256();
      auto sum3 = _mm256_setzero_si256();

      // Each innermost loop processes a 32x4 chunk of weights, so 128 weights at a time!
      constexpr int StepWidth = (simd::RegisterWidth / 8) / sizeof(int8_t);
      for (size_t j = 0; j < InSize / StepWidth; j++) {
        auto in = _mm256_load_si256(reinterpret_cast<const __m256i*>(input + j * StepWidth));
        add_dpbusd_epi32(sum0, in,
                         _mm256_load_si256(reinterpret_cast<const __m256i*>(weights[0] + offset0 + j * StepWidth)));
        add_dpbusd_epi32(sum1, in,
                         _mm256_load_si256(reinterpret_cast<const __m256i*>(weights[0] + offset1 + j * StepWidth)));
        add_dpbusd_epi32(sum2, in,
                         _mm256_load_si256(reinterpret_cast<const __m256i*>(weights[0] + offset2 + j * StepWidth)));
        add_dpbusd_epi32(sum3, in,
                         _mm256_load_si256(reinterpret_cast<const __m256i*>(weights[0] + offset3 + j * StepWidth)));
      }

      auto outval = hsumI32x4(sum0, sum1, sum2, sum3);
      outval = _mm_add_epi32(outval, _mm_loadu_si128(reinterpret_cast<const __m128i*>(biases + i * 4)));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(outputBuf + i * 4), outval);
    }
  }

  alignas(simd::Alignment) WeightType weights[OutSize][InSize];
  alignas(simd::Alignment) OutputType biases[OutSize];
  alignas(simd::Alignment) OutputType outputBuf[OutSize];
};
}  // namespace qconv::layers

#endif  // #ifndef QCONV_LAYERS_LINEAR_H_INCLUDED