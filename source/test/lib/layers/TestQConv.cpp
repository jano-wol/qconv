#include <gmock/gmock.h>

#include <cstring>

#include <layers/qconv.h>
#include <layers/qconv_naive.h>
#include <layers/qconv_thick.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::layers;
using namespace qconv::simd;
using namespace qconv::testutils;

TEST(QConv, CompareWithNaive)
{
  constexpr int SpatialIn = 16;
  constexpr int SpatialOut = 16;
  alignas(Alignment) int8_t input[SpatialIn * 20 * 20];
  alignas(Alignment) int16_t weights[SpatialIn * SpatialOut * 3 * 3];
  int32_t outputThick[SpatialOut * 20 * 20];
  QConv<SpatialIn, SpatialOut, 20, 3> q;
  QConvNaive<SpatialIn, SpatialOut, 20, 3> qN;
  QConvThick<SpatialIn, SpatialOut, 20, 3> qT;

  // random input
  randInit<int8_t>(input, SpatialIn * 20 * 20);
  randInit<int16_t>(weights, SpatialIn * SpatialOut * 3 * 3);
  q.initWeights(weights);
  q.propagate(input);
  qN.initWeights(weights);
  qN.propagate(input);
  qT.initWeights<int16_t>(weights);
  qT.propagateUSC<int8_t>(input, outputThick);
  for (int i = 0; i < SpatialOut * 20 * 20; ++i) {
    EXPECT_EQ(q.outputBuf[i], qN.outputBuf[i]);
    EXPECT_EQ(outputThick[i], qN.outputBuf[i]);
  }

  // stress high
  constInit<int8_t>(input, SpatialIn * 20 * 20, 127);
  constInit<int16_t>(weights, SpatialIn * SpatialOut * 3 * 3, 32767);
  q.initWeights(weights);
  q.propagate(input);
  qN.initWeights(weights);
  qN.propagate(input);
  qT.initWeights<int16_t>(weights);
  qT.propagateUSC<int8_t>(input, outputThick);
  for (int i = 0; i < SpatialOut * 20 * 20; ++i) {
    EXPECT_EQ(q.outputBuf[i], qN.outputBuf[i]);
    EXPECT_EQ(outputThick[i], qN.outputBuf[i]);
  }

  // stress low
  constInit<int8_t>(input, SpatialIn * 20 * 20, -128);
  constInit<int16_t>(weights, SpatialIn * SpatialOut * 3 * 3, -32768);
  q.initWeights(weights);
  q.propagate(input);
  qN.initWeights(weights);
  qN.propagate(input);
  qT.initWeights<int16_t>(weights);
  qT.propagateUSC<int8_t>(input, outputThick);
  for (int i = 0; i < SpatialOut * 20 * 20; ++i) {
    EXPECT_EQ(q.outputBuf[i], qN.outputBuf[i]);
    EXPECT_EQ(outputThick[i], qN.outputBuf[i]);
  }
}
