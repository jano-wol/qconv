#ifndef QCONV_LAYERS_QCONV_NAIVE_H_INCLUDED
#define QCONV_LAYERS_QCONV_NAIVE_H_INCLUDED

#include <fstream>

#include <layers/tile.h>

namespace qconv::layers
{
template <size_t SpatialIn, size_t SpatialOut, size_t SpatialSize, size_t KernelSize>
class QConvNaive
{
public:
  static_assert(SpatialSize == BOARDS, "Naive implementation depends on Tiles!");
  static_assert(KernelSize == 3, "Only 3x3 kernels are supported now!");
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
    WeightType w[SpatialIn * SpatialOut * KernelSize * KernelSize];
    while (ss >> curr) {
      weights[idx] = std::stof(curr);
      ++idx;
    }
    initWeights(w);
    return !stream.fail();
  }

  void initWeights(WeightType* w)
  {
    initTiles();
    for (size_t i = 0; i < SpatialIn * SpatialOut * KernelSize * KernelSize; ++i) {
      weights[i] = w[i];
    }
  }

  void propagate(InputType* input)
  {
    for (size_t i = 0; i < SpatialSize * SpatialSize; ++i) {
      for (size_t j = 0; j < SpatialOut; ++j) {
        size_t w = j * SpatialIn * KernelSize * KernelSize;
        int sum = 0;
        for (int m = 0; m < 10; ++m) {
          int g = tileAbsolute[1][i][m];
          int l = tileRelative[1][i][m];
          if (g == -1) {
            break;
          }
          for (size_t k = 0; k < SpatialIn; ++k) {
            sum += input[k * SpatialSize * SpatialSize + g] * weights[w + k * KernelSize * KernelSize + l];
          }
        }
        outputBuf[j * SpatialSize * SpatialSize + i] = sum;
      }
    }
  }

  WeightType weights[SpatialIn * SpatialOut * KernelSize * KernelSize];
  OutputBuffer outputBuf;
};
}  // namespace qconv::layers

#endif  // #ifndef QCONV_LAYERS_QCONV_NAIVE_H_INCLUDED