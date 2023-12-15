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
}

int main()
{
  testQConvNaive();
  return 0;
}
