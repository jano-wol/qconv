#include <gmock/gmock.h>

#include <cstring>

#include <layers/linear.h>
#include <layers/linear_naive.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::layers;
using namespace qconv::simd;
using namespace qconv::testutils;

TEST(Linear, CompareWithNaive)
{
  constexpr size_t InSize = 512;
  constexpr size_t OutSize = 32;
  alignas(Alignment) uint8_t input[InSize];
  alignas(Alignment) int8_t weights[OutSize][InSize];
  alignas(Alignment) int32_t biases[OutSize];
  Linear<InSize, OutSize> l;
  LinearNaive<InSize, OutSize> lN;

  // random input
  randInit<uint8_t, 0, 127>(input, InSize);
  randInit<int8_t, OutSize, InSize>(weights);
  modInit(biases, OutSize, 101);
  l.init(weights, biases);
  l.propagate(input);
  lN.init(weights, biases);
  lN.propagate(input);
  for (size_t i = 0; i < OutSize; ++i) {
    EXPECT_EQ(l.outputBuf[i], lN.outputBuf[i]);
  }

  // stress high
  constInit<uint8_t>(input, InSize, 127);
  constInit<int8_t, OutSize, InSize>(weights, 127);
  constInit(biases, OutSize, 127);
  l.init(weights, biases);
  l.propagate(input);
  lN.init(weights, biases);
  lN.propagate(input);
  for (size_t i = 0; i < OutSize; ++i) {
    EXPECT_EQ(l.outputBuf[i], lN.outputBuf[i]);
  }

  // stress low
  constInit<uint8_t>(input, InSize, 127);
  constInit<int8_t, OutSize, InSize>(weights, -128);
  constInit(biases, OutSize, -128);
  l.init(weights, biases);
  l.propagate(input);
  lN.init(weights, biases);
  lN.propagate(input);
  for (size_t i = 0; i < OutSize; ++i) {
    EXPECT_EQ(l.outputBuf[i], lN.outputBuf[i]);
  }
}
