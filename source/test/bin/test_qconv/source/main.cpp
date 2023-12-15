#include <core/utils.h>
#include <layers/qconv.h>
#include <layers/qconv_naive.h>

using namespace qconv;
using namespace qconv::core;
using namespace qconv::layers;
using namespace qconv::simdops;

void checkTest(int val, int expectedVal, const std::string& context)
{
  if (val != expectedVal) {
    std::cerr << "Test fail. val=" << val << " expectedVal=" << expectedVal << " context=" << context << "\n";
    exit(1);
  }
}

void testQConv()
{
  std::string context = "testQConv";
  std::string contextN = "testQConvNaive";
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
  checkTest(q.outputBuf[0], 64, context);
  checkTest(qN.outputBuf[0], 64, contextN);
  checkTest(qN.outputBuf[20 * 20 + 1], 96, contextN);
  checkTest(qN.outputBuf[2 * 20 * 20 + 19], 64, contextN);
  checkTest(qN.outputBuf[2 * 20 * 20 + 20], 96, contextN);
  checkTest(qN.outputBuf[3 * 20 * 20 + 21], 144, contextN);
  checkTest(qN.outputBuf[5 * 20 * 20 + 210], 144, contextN);
  checkTest(qN.outputBuf[6 * 20 * 20 + 19 * 20], 64, contextN);
  checkTest(qN.outputBuf[8 * 20 * 20 + 19 * 20 + 1], 96, contextN);
  checkTest(qN.outputBuf[15 * 20 * 20 + 20 * 20 - 1], 64, contextN);

  int16_t weightsNew[9] = {-90, -80, -70, -60, 5, 1, 2, 3, 4};
  std::memcpy(weights, weightsNew, 9 * sizeof(int16_t));
  q.initWeights(weights);
  q.propagate(input);
  qN.initWeights(weights);
  qN.propagate(input);
  checkTest(qN.outputBuf[0], 73, contextN);
  checkTest(qN.outputBuf[1], 45, contextN);
  checkTest(qN.outputBuf[19], 10, contextN);
  checkTest(qN.outputBuf[210], -150, contextN);
  checkTest(qN.outputBuf[5 * 20 * 20 + 210], 144, contextN);

  constInit(weights, SpatialIn * SpatialOut * 3 * 3, static_cast<int16_t>(1));
  std::memcpy(weights + 5 * 9, weightsNew, 9 * sizeof(int16_t));
  q.initWeights(weights);
  q.propagate(input);
  qN.initWeights(weights);
  qN.propagate(input);
  checkTest(qN.outputBuf[0], 73, contextN);
  checkTest(qN.outputBuf[5 * 20 * 20 + 210], 144, contextN);

  constInit(weights, SpatialIn * SpatialOut * 3 * 3, static_cast<int16_t>(1));
  std::memcpy(weights + 16 * 5 * 9, weightsNew, 9 * sizeof(int16_t));
  q.initWeights(weights);
  q.propagate(input);
  qN.initWeights(weights);
  qN.propagate(input);
  checkTest(qN.outputBuf[0], 64, contextN);
  checkTest(qN.outputBuf[5 * 20 * 20 + 19], 10, contextN);
  checkTest(qN.outputBuf[5 * 20 * 20 + 210], -150, contextN);

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
  checkTest(qN.outputBuf[0], -132, contextN);
  checkTest(qN.outputBuf[250], 132, contextN);
  checkTest(qN.outputBuf[20 * 20 * 10 + 250], 132, contextN);
}

int main()
{
  testQConv();
  return 0;
}
