#ifndef QCONV_LAYERS_QCONV_H_INCLUDED
#define QCONV_LAYERS_QCONV_H_INCLUDED

#include <fstream>

#include <core/simdops.h>
#include <core/utils.h>
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

  bool read_parameters(std::istream& stream)
  {
    std::string s;
    std::getline(stream, s);
    std::stringstream ss(std::move(s));
    std::string curr;
    size_t idx = 0;
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

  void initEnv(int8_t* input)
  {
    for (int i = 0; i < SpatialSize * SpatialSize; ++i) {
      if (i == 0) {
        env[i] =
            simde_mm256_setr_epi32(0, 0, 0, 0, input[i + 1], 0, input[i + SpatialSize], input[i + SpatialSize + 1]);
      } else if (i == SpatialSize - 1) {
        env[i] =
            simde_mm256_setr_epi32(0, 0, 0, input[i - 1], 0, input[i + SpatialSize - 1], input[i + SpatialSize], 0);
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
  void propagate(int8_t* input)
  {
    memset(outputBuf, 0, SpatialOut * SpatialSize * SpatialSize * sizeof(int32_t));
    for (int i = 0; i < SpatialIn; ++i) {
      initEnv(input + i * SpatialSize * SpatialSize);
      for (int j = 0; j < SpatialOut; ++j) {
        auto w = weightsEnv[j * SpatialIn + i];
        for (int k = 0; k < SpatialSize * SpatialSize; ++k) {
          auto x = simde_mm256_mullo_epi32(env[k], w);
          auto y = hsum_8x32(x);
          outputBuf[j * SpatialSize * SpatialSize + k] += y;
        }

        typedef simdops::detail::VecOp<int32_t, Inst> Op;
        typedef simdops::detail::VecLoadStore<int32_t, Alignment, Inst> LS;
        auto C = Op::set1(static_cast<int32_t>(weightsC[j * SpatialIn + i]));
        for (int b = 0; b < SpatialSize * SpatialSize / 8; ++b) {
          auto data =
              simde_mm256_setr_epi32(input[i * 400 + b * 8], input[i * 400 + b * 8 + 1], input[i * 400 + b * 8 + 2],
                                     input[i * 400 + b * 8 + 3], input[i * 400 + b * 8 + 4], input[i * 400 + b * 8 + 5],
                                     input[i * 400 + b * 8 + 6], input[i * 400 + b * 8 + 7]);
          data = simde_mm256_mullo_epi32(data, C);
          auto outputPart = LS::load(outputBuf + j * SpatialSize * SpatialSize + b * 8);
          outputPart = Op::add(data, outputPart);
          LS::store(outputBuf + j * SpatialSize * SpatialSize + b * 8, outputPart);
        }
      }
    }
  }

  alignas(Alignment) simde__m256i weightsEnv[SpatialIn * SpatialOut];
  alignas(Alignment) int16_t weightsC[SpatialIn * SpatialOut];
  alignas(Alignment) simde__m256i env[SpatialSize * SpatialSize];
  alignas(Alignment) int32_t outputBuf[SpatialOut * SpatialSize * SpatialSize];
};
}  // namespace qconv::layers

#endif  // #ifndef QCONV_LAYERS_QCONV_H_INCLUDED