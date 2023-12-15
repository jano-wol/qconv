#ifndef QCONV_LAYERS_ADD_H_INCLUDED
#define QCONV_LAYERS_ADD_H_INCLUDED

#include <algorithm>
#include <iostream>
#include <type_traits>

#include <layers/common.h>
#include <layers/tile.h>

namespace qconv::layers
{
template <IndexType SpatialOut, IndexType SpatialSize>
class AddTransform
{
public:
  using InputType = float;
  using OutputType = float;
  using OutputBuffer = OutputType[SpatialOut * SpatialSize * SpatialSize];

  // Forward propagation
  void propagate_old(InputType* input1, InputType* input2, bool update, int c)
  {
    for (size_t i = 0; i < SpatialOut * SpatialSize * SpatialSize; ++i) {
      outputBuf[i] = input1[i] + input2[i];
    }
  }

  void propagate(InputType* input1, InputType* input2, bool update, int c, int r)
  {
    if (update) {
      for (size_t i = 0; i < SpatialOut; ++i) {
        for (size_t j = 0; tileAbsolute[r][c][j] != -1; ++j) {
          int k = i * SpatialSize * SpatialSize + tileAbsolute[r][c][j];
          outputBuf[k] = input1[k] + input2[k];
        }
      }
    } else {
      for (size_t i = 0; i < SpatialOut * SpatialSize * SpatialSize; ++i) {
        outputBuf[i] = input1[i] + input2[i];
      }
    }
  }

  OutputBuffer outputBuf;
};
}  // namespace qconv::layers

#endif  // #ifndef QCONV_LAYERS_ADD_H_INCLUDED