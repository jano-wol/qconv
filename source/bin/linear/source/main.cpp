#include <cstring>

#include <layers/linear.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::layers;
using namespace qconv::simd;
using namespace qconv::testutils;

int main()
{
  alignas(Alignment) int32_t input[512];
  alignas(Alignment) int32_t weights[32][512];
  alignas(Alignment) int32_t biases[32];

  modInit(input, 512, 11);
  modInit(biases, 32, 11);
  modInit<int32_t, 32, 512>(weights, 128);
  Linear<512, 32> l;
  l.init(weights, biases);
  l.propagate(input);
  std::cout << l.outputBuf[0] << "\n";
}
