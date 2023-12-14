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

#include <core/simdops.h>
#include <layers/common.h>
#include <layers/tile.h>

namespace qconv::Layers
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
    WeightType weights[SpatialIn * SpatialOut * KernelSize * KernelSize];
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

  void initEnv(InputType* input)
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
    for (int i = 0; i < SpatialIn; i++) {
      initEnv(input + i * SpatialSize * SpatialSize);
    }
  }

  alignas(Alignment) simde__m256i weightsEnv[SpatialIn * SpatialOut];
  alignas(Alignment) WeightType weightsC[SpatialIn * SpatialOut];
  alignas(Alignment) simde__m256i env[SpatialSize * SpatialSize];
  alignas(Alignment) OutputBuffer outputBuf;
};
}  // namespace qconv::Layers

#endif  // #ifndef QCONV_LAYERS_BIAS_H_INCLUDED