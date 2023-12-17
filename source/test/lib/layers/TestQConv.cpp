#include <gmock/gmock.h>

#include <layers/qconv.h>
#include <layers/qconv_naive.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::core;
using namespace qconv::layers;
using namespace qconv::simdops;
using namespace qconv::testutils;

int getR()
{
  static int i = 1;
  int p = 10000019;

  long long q = (i % p) * (i % p);
  long long ret = q % p;
  ++i;
  return static_cast<int>(ret);
}

TEST(QConv, AllOne)
{
  constexpr int SpatialIn = 16;
  constexpr int SpatialOut = 16;
  alignas(NativeAlignment) int8_t input[SpatialIn * 20 * 20];
  alignas(NativeAlignment) int16_t weights[SpatialIn * SpatialOut * 3 * 3];
  QConv<SpatialIn, SpatialOut, 20, 3> q;
  QConvNaive<SpatialIn, SpatialOut, 20, 3> qN;
  constInit(input, SpatialIn * 20 * 20, static_cast<int8_t>(1));
  constInit(weights, SpatialIn * SpatialOut * 3 * 3, static_cast<int16_t>(1));
  q.initWeights(weights);
  q.propagate(input);
  qN.initWeights(weights);
  qN.propagate(input);
  EXPECT_EQ(q.outputBuf[0], 64);
  EXPECT_EQ(qN.outputBuf[0], 64);
  EXPECT_EQ(q.outputBuf[20 * 20 + 1], 96);
  EXPECT_EQ(qN.outputBuf[20 * 20 + 1], 96);
  EXPECT_EQ(q.outputBuf[2 * 20 * 20 + 19], 64);
  EXPECT_EQ(qN.outputBuf[2 * 20 * 20 + 19], 64);
  EXPECT_EQ(q.outputBuf[2 * 20 * 20 + 20], 96);
  EXPECT_EQ(qN.outputBuf[2 * 20 * 20 + 20], 96);
  EXPECT_EQ(q.outputBuf[3 * 20 * 20 + 21], 144);
  EXPECT_EQ(qN.outputBuf[3 * 20 * 20 + 21], 144);
  EXPECT_EQ(q.outputBuf[5 * 20 * 20 + 210], 144);
  EXPECT_EQ(qN.outputBuf[5 * 20 * 20 + 210], 144);
  EXPECT_EQ(q.outputBuf[6 * 20 * 20 + 19 * 20], 64);
  EXPECT_EQ(qN.outputBuf[6 * 20 * 20 + 19 * 20], 64);
  EXPECT_EQ(q.outputBuf[8 * 20 * 20 + 19 * 20 + 1], 96);
  EXPECT_EQ(qN.outputBuf[8 * 20 * 20 + 19 * 20 + 1], 96);
  EXPECT_EQ(q.outputBuf[15 * 20 * 20 + 20 * 20 - 1], 64);
  EXPECT_EQ(qN.outputBuf[15 * 20 * 20 + 20 * 20 - 1], 64);
}

TEST(QConv, WeightVariant1)
{
  constexpr int SpatialIn = 16;
  constexpr int SpatialOut = 16;
  alignas(NativeAlignment) int8_t input[SpatialIn * 20 * 20];
  alignas(NativeAlignment) int16_t weights[SpatialIn * SpatialOut * 3 * 3];
  QConv<SpatialIn, SpatialOut, 20, 3> q;
  QConvNaive<SpatialIn, SpatialOut, 20, 3> qN;
  constInit(input, SpatialIn * 20 * 20, static_cast<int8_t>(1));
  constInit(weights, SpatialIn * SpatialOut * 3 * 3, static_cast<int16_t>(1));
  int16_t weightsNew[9] = {-90, -80, -70, -60, 5, 1, 2, 3, 4};
  std::memcpy(weights, weightsNew, 9 * sizeof(int16_t));
  q.initWeights(weights);
  q.propagate(input);
  qN.initWeights(weights);
  qN.propagate(input);
  EXPECT_EQ(q.outputBuf[0], 73);
  EXPECT_EQ(qN.outputBuf[0], 73);
  EXPECT_EQ(q.outputBuf[1], 45);
  EXPECT_EQ(qN.outputBuf[1], 45);
  EXPECT_EQ(q.outputBuf[19], 10);
  EXPECT_EQ(qN.outputBuf[19], 10);
  EXPECT_EQ(q.outputBuf[210], -150);
  EXPECT_EQ(qN.outputBuf[210], -150);
  EXPECT_EQ(q.outputBuf[5 * 20 * 20 + 210], 144);
  EXPECT_EQ(qN.outputBuf[5 * 20 * 20 + 210], 144);
}

TEST(QConv, WeightVariant2)
{
  constexpr int SpatialIn = 16;
  constexpr int SpatialOut = 16;
  alignas(NativeAlignment) int8_t input[SpatialIn * 20 * 20];
  alignas(NativeAlignment) int16_t weights[SpatialIn * SpatialOut * 3 * 3];
  QConv<SpatialIn, SpatialOut, 20, 3> q;
  QConvNaive<SpatialIn, SpatialOut, 20, 3> qN;
  constInit(input, SpatialIn * 20 * 20, static_cast<int8_t>(1));
  constInit(weights, SpatialIn * SpatialOut * 3 * 3, static_cast<int16_t>(1));
  int16_t weightsNew[9] = {-90, -80, -70, -60, 5, 1, 2, 3, 4};
  std::memcpy(weights + 5 * 9, weightsNew, 9 * sizeof(int16_t));
  q.initWeights(weights);
  q.propagate(input);
  qN.initWeights(weights);
  qN.propagate(input);
  EXPECT_EQ(q.outputBuf[0], 73);
  EXPECT_EQ(qN.outputBuf[0], 73);
  EXPECT_EQ(q.outputBuf[5 * 20 * 20 + 210], 144);
  EXPECT_EQ(qN.outputBuf[5 * 20 * 20 + 210], 144);
}

TEST(QConv, WeightVariant3)
{
  constexpr int SpatialIn = 16;
  constexpr int SpatialOut = 16;
  alignas(NativeAlignment) int8_t input[SpatialIn * 20 * 20];
  alignas(NativeAlignment) int16_t weights[SpatialIn * SpatialOut * 3 * 3];
  QConv<SpatialIn, SpatialOut, 20, 3> q;
  QConvNaive<SpatialIn, SpatialOut, 20, 3> qN;
  constInit(input, SpatialIn * 20 * 20, static_cast<int8_t>(1));
  constInit(weights, SpatialIn * SpatialOut * 3 * 3, static_cast<int16_t>(1));
  int16_t weightsNew[9] = {-90, -80, -70, -60, 5, 1, 2, 3, 4};
  std::memcpy(weights + 16 * 5 * 9, weightsNew, 9 * sizeof(int16_t));
  q.initWeights(weights);
  q.propagate(input);
  qN.initWeights(weights);
  qN.propagate(input);
  EXPECT_EQ(q.outputBuf[0], 64);
  EXPECT_EQ(qN.outputBuf[0], 64);
  EXPECT_EQ(q.outputBuf[5 * 20 * 20 + 19], 10);
  EXPECT_EQ(qN.outputBuf[5 * 20 * 20 + 19], 10);
  EXPECT_EQ(q.outputBuf[5 * 20 * 20 + 210], -150);
  EXPECT_EQ(qN.outputBuf[5 * 20 * 20 + 210], -150);
}

TEST(QConv, InputVariant)
{
  constexpr int SpatialIn = 16;
  constexpr int SpatialOut = 16;
  alignas(NativeAlignment) int8_t input[SpatialIn * 20 * 20];
  alignas(NativeAlignment) int16_t weights[SpatialIn * SpatialOut * 3 * 3];
  QConv<SpatialIn, SpatialOut, 20, 3> q;
  QConvNaive<SpatialIn, SpatialOut, 20, 3> qN;
  constInit(input, SpatialIn * 20 * 20, static_cast<int8_t>(1));
  constInit(weights, SpatialIn * SpatialOut * 3 * 3, static_cast<int16_t>(1));
  int8_t inputLayer[20 * 20];
  int8_t val = -50;
  for (int i = 0; i < 20 * 20; ++i) {
    inputLayer[i] = val;
    if (i % 5 == 4) {
      ++val;
    }
  }
  std::memcpy(input, inputLayer, 20 * 20 * sizeof(int8_t));
  q.initWeights(weights);
  q.propagate(input);
  qN.initWeights(weights);
  qN.propagate(input);
  EXPECT_EQ(q.outputBuf[0], -132);
  EXPECT_EQ(qN.outputBuf[0], -132);
  EXPECT_EQ(q.outputBuf[250], 132);
  EXPECT_EQ(qN.outputBuf[250], 132);
  EXPECT_EQ(q.outputBuf[20 * 20 * 10 + 250], 132);
  EXPECT_EQ(qN.outputBuf[20 * 20 * 10 + 250], 132);
}

TEST(QConv, CompareWithNaive)
{
  constexpr int SpatialIn = 16;
  constexpr int SpatialOut = 16;
  alignas(NativeAlignment) int8_t input[SpatialIn * 20 * 20];
  alignas(NativeAlignment) int16_t weights[SpatialIn * SpatialOut * 3 * 3];
  QConv<SpatialIn, SpatialOut, 20, 3> q;
  QConvNaive<SpatialIn, SpatialOut, 20, 3> qN;
  for (int i = 0; i < SpatialIn * 20 * 20; ++i) {
    int r = getR();
    r = (r % 256) - 128;
    input[i] = r;
  }
  for (int i = 0; i < SpatialIn * SpatialOut * 3 * 3; ++i) {
    int r = getR();
    r = (r % 16384) - 8192;
    weights[i] = r;
  }
  q.initWeights(weights);
  q.propagate(input);
  qN.initWeights(weights);
  qN.propagate(input);
  for (int i = 0; i < SpatialOut * 20 * 20; ++i) {
    EXPECT_EQ(q.outputBuf[i], qN.outputBuf[i]);
  }
}
