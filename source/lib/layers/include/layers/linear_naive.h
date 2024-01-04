#ifndef QCONV_LAYERS_LINEAR_NAIVE_H_INCLUDED
#define QCONV_LAYERS_LINEAR_NAIVE_H_INCLUDED

#include <simd/simd.h>

namespace qconv::layers
{
template <size_t InSize, size_t OutSize>
class LinearNaive
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

  void propagate(InputType* input)
  {
    for (size_t i = 0; i < OutSize; ++i) {
      OutputType sum = biases[i];
      for (size_t j = 0; j < InSize; ++j) {
        sum += input[j] * weights[i][j];
      }
      outputBuf[i] = sum;
    }
  }

  alignas(qconv::simd::Alignment) WeightType weights[OutSize][InSize];
  alignas(qconv::simd::Alignment) OutputType biases[OutSize];
  alignas(qconv::simd::Alignment) OutputType outputBuf[OutSize];
};
}  // namespace qconv::layers

#endif  // #ifndef QCONV_LAYERS_LINEAR_NAIVE_H_INCLUDED