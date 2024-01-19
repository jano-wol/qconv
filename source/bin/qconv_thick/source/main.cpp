#include <cstring>

#include <layers/qconv_thick.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::layers;
using namespace qconv::simd;
using namespace qconv::testutils;

int main()
{
  constexpr int SpatialIn = 16;
  constexpr int SpatialOut = 16;
  alignas(simd::Alignment) int32_t input[SpatialIn * 20 * 20];
  alignas(simd::Alignment) int32_t inputPDC[SpatialIn * 22 * 22];
  alignas(simd::Alignment) int32_t output[SpatialOut * 20 * 20];
  alignas(simd::Alignment) int32_t weights[SpatialIn * SpatialOut * 3 * 3];
  modInit(input, SpatialIn * 20 * 20, 13);
  modInit(weights, SpatialIn * SpatialOut * 3 * 3, 11);
  QConvThick<SpatialIn, SpatialOut, 20, 3> q;
  q.initWeights<int32_t>(weights);
  q.USCToPDC(input, inputPDC);
  q.propagate(inputPDC);
  q.getUSCOutput(output);
  std::cout << output[0] << "\n";
}
