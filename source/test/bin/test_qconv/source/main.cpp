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

void testQConvNaive()
{
  std::string context = "testQConvNaive";
  constexpr int SpatialIn = 16;
  constexpr int SpatialOut = 16;
  alignas(NativeAlignment) int8_t input[SpatialIn * SpatialOut * 20 * 20];
  alignas(NativeAlignment) int16_t weights[SpatialIn * SpatialOut * 3 * 3];
  constInit(input, SpatialIn * SpatialOut * 20 * 20, static_cast<int8_t>(1));
  constInit(weights, SpatialIn * SpatialOut * 3 * 3, static_cast<int16_t>(1));
  QConvNaive<SpatialIn, SpatialOut, 20, 3> q;

  q.initWeights(weights);
  q.propagate(input);
  checkTest(q.outputBuf[0], 64, context);
  checkTest(q.outputBuf[20 * 20 + 1], 96, context);
  checkTest(q.outputBuf[2 * 20 * 20 + 19], 64, context);
  checkTest(q.outputBuf[2 * 20 * 20 + 20], 96, context);
  checkTest(q.outputBuf[3 * 20 * 20 + 21], 144, context);
  checkTest(q.outputBuf[5 * 20 * 20 + 210], 144, context);
  checkTest(q.outputBuf[6 * 20 * 20 + 19 * 20], 64, context);
  checkTest(q.outputBuf[8 * 20 * 20 + 19 * 20 + 1], 96, context);
  checkTest(q.outputBuf[15 * 20 * 20 + 20 * 20 - 1], 64, context);

  int16_t weightsNew[9] = {-90, -80, -70, -60, 5, 1, 2, 3, 4};
  std::memcpy(weights, weightsNew, 9 * sizeof(int16_t));
  q.initWeights(weights);
  q.propagate(input);
  checkTest(q.outputBuf[0], 73, context);
  checkTest(q.outputBuf[1], 45, context);
  checkTest(q.outputBuf[19], 10, context);
  checkTest(q.outputBuf[210], -150, context);
  checkTest(q.outputBuf[5 * 20 * 20 + 210], 144, context);
}

int main()
{
  testQConvNaive();
  return 0;
}
