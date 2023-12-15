#ifndef QCONV_LAYERS_BIAS_H_INCLUDED
#define QCONV_LAYERS_BIAS_H_INCLUDED

#include <algorithm>
#include <iostream>
#include <type_traits>

#include <layers/common.h>
#include <layers/tile.h>

namespace qconv::layers
{
template <IndexType InSize, IndexType SpatialSize>
class BiasTransform
{
public:
  using InputType = float;
  using OutputType = float;
  using OutputBuffer = OutputType[InSize * SpatialSize * SpatialSize];

  bool read_parameters(std::istream& stream)
  {
    std::string s;
    std::getline(stream, s);
    std::stringstream ss(std::move(s));
    std::string curr;
    size_t idx = 0;
    while (ss >> curr) {
      weights[idx] = std::stof(curr);
      ++idx;
    }
    return !stream.fail();
  }

  // Forward propagation
  void propagate(InputType* input, bool update, int c)
  {
    for (size_t i = 0; i < InSize; ++i) {
      WeightType d = weights[i];
      for (size_t j = 0; j < SpatialSize * SpatialSize; ++j) {
        outputBuf[i * SpatialSize * SpatialSize + j] = input[i * SpatialSize * SpatialSize + j] + d;
      }
    }
  }

  using WeightType = float;
  WeightType weights[InSize];
  OutputBuffer outputBuf;
};
}  // namespace qconv::layers

#endif  // #ifndef QCONV_LAYERS_BIAS_H_INCLUDED