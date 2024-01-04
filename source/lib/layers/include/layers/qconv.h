#ifndef QCONV_LAYERS_QCONV_H_INCLUDED
#define QCONV_LAYERS_QCONV_H_INCLUDED

#include <fstream>
#include <sstream>

#include <layers/tile.h>
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
template <uint32_t SpatialIn, uint32_t SpatialOut, uint32_t SpatialSize, uint32_t KernelSize>
class QConv
{
public:
  static_assert(KernelSize == 3, "Only 3x3 kernels are supported now!");

  bool read_parameters(std::istream& stream)
  {
    std::string s;
    std::getline(stream, s);
    std::stringstream ss(std::move(s));
    std::string curr;
    uint32_t idx = 0;
    int16_t w[SpatialIn * SpatialOut * KernelSize * KernelSize];
    while (ss >> curr) {
      w[idx] = std::stof(curr);
      ++idx;
    }
    initWeights(w);
    return !stream.fail();
  }

  void initWeights(int16_t* w)
  {
    uint32_t weightsEnvIdx = 0;
    uint32_t weightsCIdx = 0;
    for (uint32_t i = 0; i < SpatialIn * SpatialOut * KernelSize * KernelSize; i += KernelSize * KernelSize) {
      weightsEnv[weightsEnvIdx] =
          _mm256_setr_epi32(w[i + 0], w[i + 1], w[i + 2], w[i + 3], w[i + 5], w[i + 6], w[i + 7], w[i + 8]);
      ++weightsEnvIdx;
      weightsC[weightsCIdx] = w[i + 4];
      ++weightsCIdx;
    }
  }

  void initEnv(int8_t* input)
  {
    for (uint32_t i = 0; i < SpatialSize * SpatialSize; ++i) {
      if (i == 0) {
        env[i] = _mm256_setr_epi32(0, 0, 0, 0, input[i + 1], 0, input[i + SpatialSize], input[i + SpatialSize + 1]);
      } else if (i == SpatialSize - 1) {
        env[i] = _mm256_setr_epi32(0, 0, 0, input[i - 1], 0, input[i + SpatialSize - 1], input[i + SpatialSize], 0);
      } else if (i == SpatialSize * (SpatialSize - 1)) {
        env[i] = _mm256_setr_epi32(0, input[i - SpatialSize], input[i - SpatialSize + 1], 0, input[i + 1], 0, 0, 0);
      } else if (i == SpatialSize * SpatialSize - 1) {
        env[i] = _mm256_setr_epi32(input[i - SpatialSize - 1], input[i - SpatialSize], 0, input[i - 1], 0, 0, 0, 0);
      } else if (i < SpatialSize) {
        env[i] = _mm256_setr_epi32(0, 0, 0, input[i - 1], input[i + 1], input[i + SpatialSize - 1],
                                   input[i + SpatialSize], input[i + SpatialSize + 1]);
      } else if (i % SpatialSize == 0) {
        env[i] = _mm256_setr_epi32(0, input[i - SpatialSize], input[i - SpatialSize + 1], 0, input[i + 1], 0,
                                   input[i + SpatialSize], input[i + SpatialSize + 1]);
      } else if (i % SpatialSize == (SpatialSize - 1)) {
        env[i] = _mm256_setr_epi32(input[i - SpatialSize - 1], input[i - SpatialSize], 0, input[i - 1], 0,
                                   input[i + SpatialSize - 1], input[i + SpatialSize], 0);
      } else if (i > SpatialSize * (SpatialSize - 1)) {
        env[i] = _mm256_setr_epi32(input[i - SpatialSize - 1], input[i - SpatialSize], input[i - SpatialSize + 1],
                                   input[i - 1], input[i + 1], 0, 0, 0);
      } else {
        env[i] = _mm256_setr_epi32(input[i - SpatialSize - 1], input[i - SpatialSize], input[i - SpatialSize + 1],
                                   input[i - 1], input[i + 1], input[i + SpatialSize - 1], input[i + SpatialSize],
                                   input[i + SpatialSize + 1]);
      }
    }
  }

  // Forward propagation
  void propagate(int8_t* input)
  {
    memset(outputBuf, 0, SpatialOut * SpatialSize * SpatialSize * sizeof(int32_t));
    for (uint32_t i = 0; i < SpatialIn; ++i) {
      initEnv(input + i * SpatialSize * SpatialSize);
      for (uint32_t j = 0; j < SpatialOut; ++j) {
        auto w = weightsEnv[j * SpatialIn + i];
        for (uint32_t k = 0; k < SpatialSize * SpatialSize; ++k) {
          auto x = _mm256_mullo_epi32(env[k], w);
          auto y = hsum_8x32(x);
          outputBuf[j * SpatialSize * SpatialSize + k] += y;
        }

        constexpr int StepWidth = (simd::RegisterWidth / 8) / sizeof(int32_t);
        __m128i c = _mm_set1_epi32(static_cast<int32_t>(weightsC[j * SpatialIn + i]));
        __m256i C = _mm256_set_m128i(c, c);
        for (uint32_t b = 0; b < SpatialSize * SpatialSize / StepWidth; ++b) {
          int inIdx = i * SpatialSize * SpatialSize + b * StepWidth;
          int outIdx = j * SpatialSize * SpatialSize + b * StepWidth;
          auto data = _mm256_setr_epi32(input[inIdx], input[inIdx + 1], input[inIdx + 2], input[inIdx + 3],
                                        input[inIdx + 4], input[inIdx + 5], input[inIdx + 6], input[inIdx + 7]);
          data = _mm256_mullo_epi32(data, C);
          auto outputPart = _mm256_load_si256(reinterpret_cast<const __m256i*>(outputBuf + outIdx));
          outputPart = _mm256_add_epi32(data, outputPart);
          _mm256_store_si256(reinterpret_cast<__m256i*>(outputBuf + outIdx), outputPart);
        }
      }
    }
  }

  alignas(simd::Alignment) __m256i weightsEnv[SpatialIn * SpatialOut];
  alignas(simd::Alignment) int16_t weightsC[SpatialIn * SpatialOut];
  alignas(simd::Alignment) __m256i env[SpatialSize * SpatialSize];
  alignas(simd::Alignment) int32_t outputBuf[SpatialOut * SpatialSize * SpatialSize];
};
}  // namespace qconv::layers

#endif  // #ifndef QCONV_LAYERS_QCONV_H_INCLUDED