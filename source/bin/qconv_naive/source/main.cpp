#include <cstring>

#include <layers/qconv_naive.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::layers;
using namespace qconv::simd;
using namespace qconv::testutils;

int main()
{
  constexpr int SpatialIn = 16;
  constexpr int SpatialOut = 16;
  alignas(simd::Alignment) int8_t input[SpatialIn * 20 * 20];
  alignas(simd::Alignment) int16_t weights[SpatialIn * SpatialOut * 3 * 3];
  modInit(input, SpatialIn * 20 * 20, 13);
  modInit(weights, SpatialIn * SpatialOut * 3 * 3, 11);
  QConvNaive<SpatialIn, SpatialOut, 20, 3> q;
  q.initWeights(weights);
  q.propagate(input);
  std::cout << q.outputBuf[0] << "\n";
}
