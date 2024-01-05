#include <cstring>

#include <benchutils/benchutils.h>
#include <layers/linear.h>
#include <layers/linear_naive.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::layers;
using namespace qconv::simd;
using namespace qconv::testutils;

void linear_naive(benchmark::State& state)
{
  alignas(Alignment) int8_t input[512];
  alignas(Alignment) int8_t weights[32][512];
  alignas(Alignment) int32_t biases[32];

  modInit(input, 512, 11);
  modInit(biases, 32, 11);
  modInit<int8_t, 32, 512>(weights, 128);
  LinearNaive<512, 32> l;
  l.init(weights, biases);
  for (auto _ : state) {
    l.propagate(input);
    benchmark::ClobberMemory();
  }
  checkTrue(l.outputBuf[0] == 161802);
  checkTrue(l.outputBuf[1] == 161532);
  checkTrue(l.outputBuf[30] == 162384);
  checkTrue(l.outputBuf[31] == 161986);
}
BENCHMARK(linear_naive);

void linear_simdops(benchmark::State& state)
{
  alignas(Alignment) int8_t input[512];
  alignas(Alignment) int8_t weights[32][512];
  alignas(Alignment) int32_t biases[32];

  modInit(input, 512, 11);
  modInit(biases, 32, 11);
  modInit<int8_t, 32, 512>(weights, 128);
  Linear<512, 32> l;
  l.init(weights, biases);
  for (auto _ : state) {
    l.propagate(input);
    benchmark::ClobberMemory();
  }
  checkTrue(l.outputBuf[0] == 161802);
  checkTrue(l.outputBuf[1] == 161532);
  checkTrue(l.outputBuf[30] == 162384);
  checkTrue(l.outputBuf[31] == 161986);
}
BENCHMARK(linear_simdops);
BENCHMARK_MAIN();
