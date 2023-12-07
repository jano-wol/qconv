/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2022 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef QCONV_LAYERS_BIAS_H_INCLUDED
#define QCONV_LAYERS_BIAS_H_INCLUDED

#include <algorithm>
#include <iostream>
#include <type_traits>

#include <layers/common.h>
#include <layers/tile.h>

namespace qconv::Layers
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
}  // namespace qconv::Layers

#endif  // #ifndef QCONV_LAYERS_BIAS_H_INCLUDED