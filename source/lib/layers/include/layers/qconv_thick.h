#ifndef QCONV_LAYERS_QCONV2_H_INCLUDED
#define QCONV_LAYERS_QCONV2_H_INCLUDED

#include <cassert>
#include <cstring>

#include <simd/simd.h>

namespace qconv::layers
{
template <size_t SpatialIn, size_t SpatialOut, size_t SpatialSize, size_t KernelSize>
class QConvThick
{
public:
  static_assert(KernelSize == 3, "Only 3x3 kernels are supported now!");
  static_assert(SpatialIn % 8 == 0 && SpatialOut % 8 == 0, "Only eigth divisible spatial dims are supported now!");
  using InputType = int32_t;
  using WeightType = int32_t;
  using OutputType = int32_t;
  static constexpr size_t SpatialSizePadded = SpatialSize + 2;
  static constexpr size_t InStep = SpatialIn / 8;
  static constexpr size_t OutStep = SpatialOut / 8;

  template <typename T>
  WeightType getW(int in, int out, int kernelId, T* w)
  {
    auto ret = static_cast<WeightType>(w[out * SpatialIn * KernelSize * KernelSize + in * KernelSize * KernelSize +
                                         KernelSize * KernelSize - 1 - kernelId]);
    return ret;
  }

  template <typename T>
  void initWeights(T* w)
  {
    for (size_t k = 0; k < InStep; ++k) {
      for (size_t l = 0; l < OutStep; ++l) {
        for (size_t m = 0; m < KernelSize * KernelSize; ++m) {
          for (size_t n = 0; n < 8; ++n) {
            auto w1 = getW<T>(k * 8 + (n + 0) % 8, l * 8 + 0, m, w);
            auto w2 = getW<T>(k * 8 + (n + 1) % 8, l * 8 + 1, m, w);
            auto w3 = getW<T>(k * 8 + (n + 2) % 8, l * 8 + 2, m, w);
            auto w4 = getW<T>(k * 8 + (n + 3) % 8, l * 8 + 3, m, w);
            auto w5 = getW<T>(k * 8 + (n + 4) % 8, l * 8 + 4, m, w);
            auto w6 = getW<T>(k * 8 + (n + 5) % 8, l * 8 + 5, m, w);
            auto w7 = getW<T>(k * 8 + (n + 6) % 8, l * 8 + 6, m, w);
            auto w8 = getW<T>(k * 8 + (n + 7) % 8, l * 8 + 7, m, w);
            weights[k][l][m][n] = _mm256_setr_epi32(w1, w2, w3, w4, w5, w6, w7, w8);
          }
        }
      }
    }
  }

  bool isPad(size_t i)
  {
    return ((i < SpatialSizePadded) || (i % SpatialSizePadded == 0) || ((i + 1) % SpatialSizePadded == 0) ||
            (SpatialSizePadded * (SpatialSizePadded - 1) <= i));
  }

  // Forward propagation
  void propagate(InputType* input)
  {
    assert(simd::isPtrAligned(input));
    std::memset(outputBuf, 0, SpatialOut * SpatialSizePadded * SpatialSizePadded * sizeof(OutputType));
    const auto input256 = reinterpret_cast<const __m256i*>(input);
    auto output256 = reinterpret_cast<__m256i*>(outputBuf);
    int s = static_cast<int>(SpatialSizePadded);
    const int relDir[KernelSize * KernelSize] = {-s - 1, -s, -s + 1, -1, 0, 1, s - 1, s, s + 1};
    const __m256i epi32_256_ctl_1 = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);
    for (size_t i = 0; i < SpatialSize; ++i) {
      for (size_t j = 0; j < SpatialSize; ++j) {
        size_t InIdx = (i + 1) * SpatialSizePadded + (j + 1);
        for (size_t k = 0; k < InStep; ++k) {
          __m256i curr = input256[InIdx * InStep + k];
          for (size_t l = 0; l < OutStep; ++l) {
            for (size_t m = 0; m < KernelSize * KernelSize; ++m) {
              size_t OutIdx = InIdx + relDir[m];
              __m256i& update = output256[OutIdx * OutStep + l];
              for (size_t n = 0; n < 8; ++n) {
                auto mul = _mm256_mullo_epi32(curr, weights[k][l][m][n]);
                update = _mm256_add_epi32(mul, update);
                curr = _mm256_permutevar8x32_epi32(curr, epi32_256_ctl_1);
              }
            }
          }
        }
      }
    }
  }

  template <typename T>
  void padInput(T* input, InputType* paddedInput)
  {
    InputType help[SpatialIn * SpatialSizePadded * SpatialSizePadded];
    size_t inIdx = 0;
    for (size_t i = 0; i < SpatialIn; ++i) {
      for (size_t j = 0; j < SpatialSizePadded * SpatialSizePadded; ++j) {
        if (isPad(j)) {
          help[i * SpatialSizePadded * SpatialSizePadded + j] = -1;
        } else {
          help[i * SpatialSizePadded * SpatialSizePadded + j] = static_cast<InputType>(input[inIdx++]);
        }
      }
    }
    for (size_t i = 0; i < SpatialIn * SpatialSizePadded * SpatialSizePadded; ++i) {
      auto pos = i / SpatialIn;
      auto depth = i % SpatialIn;
      paddedInput[i] = help[depth * SpatialSizePadded * SpatialSizePadded + pos];
    }
  }

  void getUnpaddedOutput(OutputType* output)
  {
    for (size_t i = 0; i < SpatialOut; ++i) {
      for (size_t j = 0; j < SpatialSize * SpatialSize; ++j) {
        auto pos = (j / SpatialSize + 1) * SpatialSizePadded + j % SpatialSize + 1;
        output[i * SpatialSize * SpatialSize + j] = outputBuf[pos * SpatialOut + i];
      }
    }
  }

  template <typename T>
  void propagateRaw(T* input, OutputType* output)
  {
    alignas(simd::Alignment) InputType paddedInput[SpatialIn * SpatialSizePadded * SpatialSizePadded];
    padInput<T>(input, paddedInput);
    propagate(paddedInput);
    getUnpaddedOutput(output);
  }

  alignas(simd::Alignment) __m256i weights[InStep][OutStep][KernelSize * KernelSize][8];
  alignas(simd::Alignment) OutputType outputBuf[SpatialOut * SpatialSizePadded * SpatialSizePadded];
};
}  // namespace qconv::layers

#endif  // #ifndef QCONV_LAYERS_QCONV2_H_INCLUDED