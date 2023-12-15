
#include <core/utils.h>
#include <layers/qconv.h>

using namespace qconv;
using namespace qconv::core;
using namespace qconv::layers;
using namespace qconv::simdops;

void test1()
{
  constexpr int SpatialIn = 16;
  constexpr int SpatialOut = 16;
  alignas(NativeAlignment) int8_t input[SpatialIn * SpatialOut * 20 * 20];
  alignas(NativeAlignment) int16_t weights[SpatialIn * SpatialOut * 3 * 3];
  modInit(input, SpatialIn * SpatialOut * 20 * 20, 13);
  modInit(weights, SpatialIn * SpatialOut * 3 * 3, 11);
  QConv<SpatialIn, SpatialOut, 20, 3> q;
  q.initWeights(weights);
}

int main()
{
  test1();
  return 0;
}
