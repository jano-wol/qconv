#include <gmock/gmock.h>

#include <cstring>

#include <layers/linear.h>
#include <layers/linear_naive.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::layers;
using namespace qconv::simd;
using namespace qconv::testutils;

TEST(Linear, AllOne)
{
  constexpr int InSize = 512;
  constexpr int OutSize = 32;
  alignas(Alignment) int8_t input[InSize];
  alignas(Alignment) int8_t weights[InSize * OutSize];
  alignas(Alignment) int32_t biases[OutSize];
  Linear<InSize, OutSize> l;
  LinearNaive<InSize, OutSize> lN;
  constInit(input, InSize, static_cast<int8_t>(1));
  constInit(weights, InSize * OutSize, static_cast<int8_t>(1));
  constInit(biases, OutSize, static_cast<int32_t>(1));
  l.init(weights, biases);
  l.propagate(input);
  lN.init(weights, biases);
  lN.propagate(input);
  EXPECT_EQ(l.outputBuf[0], 513);
  EXPECT_EQ(lN.outputBuf[0], 513);
  EXPECT_EQ(l.outputBuf[1], 513);
  EXPECT_EQ(lN.outputBuf[1], 513);
}
