#ifndef QCONV_LAYERS_QCONV2_H_INCLUDED
#define QCONV_LAYERS_QCONV2_H_INCLUDED

#include <cassert>
#include <cstring>

#include <simd/simd.h>

namespace qconv::layers
{
template <size_t SpatialIn, size_t SpatialOut, size_t SpatialSize, size_t KernelSize>
class QConv2
{
public:
  static_assert(KernelSize == 3, "Only 3x3 kernels are supported now!");
  static_assert(SpatialIn % 8 == 0 && SpatialOut % 8 == 0, "Only eigth divisible spatial dims are supported now!");
  using InputType = int32_t;
  using WeightType = int32_t;
  using OutputType = int32_t;
  constexpr size_t SpatialSizePadded = SpatialSize + 2;
  constexpr size_t InStep = SpatialIn / 8;
  constexpr size_t OutStep = SpatialOut / 8;

  WeightType getW(int in, int out, int kernelId, WeightType* w)
  {
    return w[out * SpatialIn * KernelSize * KernelSize + in * KernelSize * KernelSize + kenelId];
  }

  void initWeights(WeightType* w)
  {
    for (size_t k = 0; k < InStep; ++k) {
      for (size_t l = 0; l < OutStep; ++l) {
        for (size_t m = 0; m < KernelSize * KernelSize; ++m) {
          for (size_t n = 0; n < 8; ++n) {
            auto w1 = getW(k * 8 + (n + 0) % 8, l * 8 + 0, m, w);
            auto w2 = getW(k * 8 + (n + 1) % 8, l * 8 + 1, m, w);
            auto w3 = getW(k * 8 + (n + 2) % 8, l * 8 + 2, m, w);
            auto w4 = getW(k * 8 + (n + 3) % 8, l * 8 + 3, m, w);
            auto w5 = getW(k * 8 + (n + 4) % 8, l * 8 + 4, m, w);
            auto w6 = getW(k * 8 + (n + 5) % 8, l * 8 + 5, m, w);
            auto w7 = getW(k * 8 + (n + 6) % 8, l * 8 + 6, m, w);
            auto w8 = getW(k * 8 + (n + 7) % 8, l * 8 + 7, m, w);
            weights[k][l][m][n] = _mm256_setr_epi32(w1, w2, w3, w4, w5, w6, w7, w8);
          }
        }
      }
    }
  }

  // Forward propagation
  void propagate(InputType* input)
  {
    assert(simd::isPtrAligned(input));
    std::memset(outputBuf, 0, SpatialOut * SpatialSizePadded * SpatialSizePadded * sizeof(OutputType));
    const auto input256 = reinterpret_cast<const __mm256i*>(input);
    auto output256 = reinterpret_cast<__mm256i*>(outputBuf);
    const int relDir[KernelSize * KernelSize] = {
        -SpatialSizePadded - 1, -SpatialSizePadded, -SpatialSizePadded + 1, -1, 0, 1,
        SpatialSizePadded - 1,  SpatialSizePadded,  SpatialSizePadded + 1};
    const __m256i epi32_256_ctl_1 = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    for (size_t i = 0; i < SpatialSize; ++i) {
      for (size_t j = 0; j < SpatialSize; ++j) {
        size_t InIdx = (i + 1) * SpatialSizePadded + (j + 1);
        for (size_t k = 0; k < InStep; ++k) {
          __mm256i curr = input256[InIdx * InStep + k];
          for (size_t l = 0; l < OutputStep; ++l) {
            for (size_t m = 0; m < KernelSize * KernelSize; ++m) {
              size_t OutIdx = InIdx + relDir[m];
              for (size_t n = 0; n < 8; ++n) {
                auto mul = _mm256_mullo_epi32(curr, weights[k][l][m][n]);
                output256[OutIdx * OutStep + l] = _mm256_add_epi32(mul, output256[OutIdx * OutStep + l]);
                curr = _mm256_permutevar8x32_epi32(curr, epi32_256_ctl_1);
              }
            }
          }
        }
      }
    }
  }

  alignas(simd::Alignment) __m256i weights[InStep][OutStep][KernelSize * KernelSize][8];
  alignas(simd::Alignment) OutputType outputBuf[SpatialOut * SpatialSizePadded * SpatialSizePadded];
};
}  // namespace qconv::layers

#endif  // #ifndef QCONV_LAYERS_QCONV2_H_INCLUDED