#ifndef QCONV_LAYERS_QCONV_NAIVE_H_INCLUDED
#define QCONV_LAYERS_QCONV_NAIVE_H_INCLUDED

#include <fstream>
#include <sstream>

#include <simd/simd.h>

namespace qconv::layers
{
template <size_t SpatialIn, size_t SpatialOut, size_t SpatialSize, size_t KernelSize>
class QConvNaive
{
public:
  static_assert(KernelSize == 3, "Only 3x3 kernels are supported now!");
  using InputType = int8_t;
  using OutputType = int32_t;
  using WeightType = int16_t;
  using OutputBuffer = OutputType[SpatialOut * SpatialSize * SpatialSize];

  void initTiles()
  {
    for (int k = 0; k < 7; ++k) {
      for (int c = 0; c < static_cast<int>(SpatialSize * SpatialSize); ++c) {
        int cMod = c % SpatialSize;
        int locIdx = 0;
        int idx = 0;
        tileAbsolute[k][c][idx] = -1;
        tileRelative[k][c][idx] = -1;
        for (int i = -k; i <= k; ++i) {
          for (int j = -k; j <= k; ++j) {
            int curr = c + i * SpatialSize + j;
            if (curr >= 0 && curr < static_cast<int>(SpatialSize * SpatialSize)) {
              if (j == 0) {
                tileAbsolute[k][c][idx] = curr;
                tileRelative[k][c][idx] = locIdx;
                ++idx;
                tileAbsolute[k][c][idx] = -1;
                tileRelative[k][c][idx] = -1;
              }
              if (j > 0 && static_cast<int>(curr % SpatialSize) > cMod) {
                tileAbsolute[k][c][idx] = curr;
                tileRelative[k][c][idx] = locIdx;
                ++idx;
                tileAbsolute[k][c][idx] = -1;
                tileRelative[k][c][idx] = -1;
              }
              if (j < 0 && static_cast<int>(curr % SpatialSize) < cMod) {
                tileAbsolute[k][c][idx] = curr;
                tileRelative[k][c][idx] = locIdx;
                ++idx;
                tileAbsolute[k][c][idx] = -1;
                tileRelative[k][c][idx] = -1;
              }
            }
            ++locIdx;
          }
        }
      }
    }
  }

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

  alignas(qconv::simd::Alignment) int tileAbsolute[7][SpatialSize * SpatialSize][15 * 15 + 1];
  alignas(qconv::simd::Alignment) int tileRelative[7][SpatialSize * SpatialSize][15 * 15 + 1];
  alignas(qconv::simd::Alignment) WeightType weights[SpatialIn * SpatialOut * KernelSize * KernelSize];
  OutputBuffer outputBuf;
};
}  // namespace qconv::layers

#endif  // #ifndef QCONV_LAYERS_QCONV_NAIVE_H_INCLUDED