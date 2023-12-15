#ifndef QCONV_LAYERS_QCONV_H_INCLUDED
#define QCONV_LAYERS_QCONV_H_INCLUDED

#include <algorithm>
#include <iostream>

#include <core/simdops.h>
#include <layers/common.h>
#include <layers/tile.h>

static inline int32_t hsum_epi32(simde__m128i x)
{
  simde__m128i hi64 =
      simde_mm_unpackhi_epi64(x, x);  // 3-operand non-destructive AVX lets us save a byte without needing a movdqa
  simde__m128i sum64 = simde_mm_add_epi32(hi64, x);
  simde__m128i hi32 = simde_mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));  // Swap the low two elements
  simde__m128i sum32 = simde_mm_add_epi32(sum64, hi32);
  return simde_mm_cvtsi128_si32(sum32);  // movd
}

static inline int32_t hsum_8x32(simde__m256i v)
{
  __m128i sum128 = simde_mm_add_epi32(
      simde_mm256_castsi256_si128(v),
      simde_mm256_extracti128_si256(v, 1));  // silly GCC uses a longer AXV512VL instruction if AVX512 is enabled :/
  return hsum_epi32(sum128);
}

namespace qconv::layers
{
template <IndexType SpatialIn, IndexType SpatialOut, IndexType SpatialSize, IndexType KernelSize,
          int Alignment = simdops::NativeAlignment, simdops::InstructionType Inst = simdops::NativeInstType>
class QConv
{
public:
  static_assert(Inst == simdops::InstructionType::AVX2, "Only avx2 is supported now!");
  static_assert(KernelSize == 3, "Only 3x3 kernels are supported now!");
  static_assert(simdops::isAlignSizeOK(Alignment));
  using InputType = int8_t;
  using OutputType = int32_t;
  using WeightType = int16_t;
  using OutputBuffer = OutputType[SpatialOut * SpatialSize * SpatialSize];

  bool read_parameters(std::istream& stream)
  {
    std::string s;
    std::getline(stream, s);
    std::stringstream ss(std::move(s));
    std::string curr;
    size_t idx = 0;
    // WeightType weights[SpatialIn * SpatialOut * KernelSize * KernelSize];
    while (ss >> curr) {
      weights[idx] = std::stof(curr);
      ++idx;
    }
    initWeights(weights);
    return !stream.fail();
  }

  void initWeights(WeightType* w)
  {
    size_t weightsEnvIdx = 0;
    size_t weightsCIdx = 0;
    for (int i = 0; i < SpatialIn * SpatialOut * KernelSize * KernelSize; i += KernelSize * KernelSize) {
      weightsEnv[weightsEnvIdx] =
          simde_mm256_setr_epi32(w[i + 0], w[i + 1], w[i + 2], w[i + 3], w[i + 5], w[i + 6], w[i + 7], w[i + 8]);
      ++weightsEnvIdx;
      weightsC[weightsCIdx] = w[i + 4];
      ++weightsCIdx;
    }
  }

  void initWeightsNaive(WeightType* w)
  {
    for (int i = 0; i < SpatialIn * SpatialOut * KernelSize * KernelSize; ++i) {
      weights[i] = w[i];
    }
  }

  void inline initEnv(InputType* input)
  {
    for (int i = 0; i < SpatialSize * SpatialSize; ++i) {
      if (i == 0) {
        env[i] =
            simde_mm256_setr_epi32(0, 0, 0, 0, input[i + 1], 0, input[i + SpatialSize], input[i + SpatialSize + 1]);
      } else if (i == SpatialSize - 1) {
        env[i] = simde_mm256_setr_epi32(0, 0, 0, input[i - 1], 0, input[i + SpatialSize - 1], input[i + SpatialSize],
                                        input[i + SpatialSize + 1]);
      } else if (i == SpatialSize * (SpatialSize - 1)) {
        env[i] =
            simde_mm256_setr_epi32(0, input[i - SpatialSize], input[i - SpatialSize + 1], 0, input[i + 1], 0, 0, 0);
      } else if (i == SpatialSize * SpatialSize - 1) {
        env[i] =
            simde_mm256_setr_epi32(input[i - SpatialSize - 1], input[i - SpatialSize], 0, input[i - 1], 0, 0, 0, 0);
      } else if (i < SpatialSize) {
        env[i] = simde_mm256_setr_epi32(0, 0, 0, input[i - 1], input[i + 1], input[i + SpatialSize - 1],
                                        input[i + SpatialSize], input[i + SpatialSize + 1]);
      } else if (i % SpatialSize == 0) {
        env[i] = simde_mm256_setr_epi32(0, input[i - SpatialSize], input[i - SpatialSize + 1], 0, input[i + 1], 0,
                                        input[i + SpatialSize], input[i + SpatialSize + 1]);
      } else if (i % SpatialSize == (SpatialSize - 1)) {
        env[i] = simde_mm256_setr_epi32(input[i - SpatialSize - 1], input[i - SpatialSize], 0, input[i - 1], 0,
                                        input[i + SpatialSize - 1], input[i + SpatialSize], 0);
      } else if (i > SpatialSize * (SpatialSize - 1)) {
        env[i] = simde_mm256_setr_epi32(input[i - SpatialSize - 1], input[i - SpatialSize], input[i - SpatialSize + 1],
                                        input[i - 1], input[i + 1], 0, 0, 0);
      } else {
        env[i] = simde_mm256_setr_epi32(input[i - SpatialSize - 1], input[i - SpatialSize], input[i - SpatialSize + 1],
                                        input[i - 1], input[i + 1], input[i + SpatialSize - 1], input[i + SpatialSize],
                                        input[i + SpatialSize + 1]);
      }
    }
  }

  // Forward propagation
  void propagate(InputType* input)
  {
    memset(outputBuf, 0, SpatialOut * SpatialSize * SpatialSize * sizeof(OutputType));
    for (int i = 0; i < SpatialIn; ++i) {
      initEnv(input + i * SpatialSize * SpatialSize);
      for (int j = 0; j < SpatialOut; ++j) {
        auto w = weightsEnv[i * SpatialOut + j];
        for (int k = 0; k < SpatialSize * SpatialSize; ++k) {
          auto x = simde_mm256_mullo_epi32(env[k], w);
          auto y = hsum_8x32(x);
          outputBuf[j * SpatialSize * SpatialSize + k] += y;
        }
      }
    }
  }

  void inline propagateNaive(InputType* input)
  {
    for (size_t i = 0; i < SpatialSize * SpatialSize; ++i) {
      for (size_t j = 0; j < SpatialOut; ++j) {
        size_t w = j * SpatialIn * KernelSize * KernelSize;
        int sum = 0;
        for (int m = 0; m < 10; ++m) {
          int g = conv_global[1][i][m];
          int l = conv_rel_global[1][i][m];
          if (g == -1) {
            break;
          }
          for (int k = 0; k < SpatialIn; ++k) {
            sum += input[k * SpatialSize * SpatialSize + g] * weights[w + k * KernelSize * KernelSize + l];
          }
        }
        outputBuf[j * SpatialSize * SpatialSize + i] = sum;
      }
    }
  }

  alignas(Alignment) WeightType weights[SpatialIn * SpatialOut * KernelSize * KernelSize];
  alignas(Alignment) simde__m256i weightsEnv[SpatialIn * SpatialOut];
  alignas(Alignment) WeightType weightsC[SpatialIn * SpatialOut];
  alignas(Alignment) simde__m256i env[SpatialSize * SpatialSize];
  alignas(Alignment) OutputBuffer outputBuf;
};
}  // namespace qconv::layers

#endif  // #ifndef QCONV_LAYERS_QCONV_H_INCLUDED