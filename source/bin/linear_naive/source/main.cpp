#include <cstring>

#include <layers/linear_naive.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::layers;
using namespace qconv::simd;
using namespace qconv::testutils;

int main()
{
  alignas(Alignment) int8_t input[512];
  alignas(Alignment) int8_t weights[32][512];
  alignas(Alignment) int32_t biases[32];

  modInit(input, 512, 11);
  modInit(biases, 32, 11);
  modInit<int8_t, 32, 512>(weights, 128);
  LinearNaive<512, 32> lN;
  lN.init(weights, biases);
  lN.propagate(input);
  std::cout << lN.outputBuf[0] << "\n";
}
