#include <cstring>

#include <benchutils/benchutils.h>
#include <layers/qconv.h>
#include <layers/qconv_naive.h>
#include <layers/qconv_thick.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::layers;
using namespace qconv::testutils;

void qconv_naive(benchmark::State& state)
{
  constexpr int SpatialIn = 16;
  constexpr int SpatialOut = 16;
  alignas(simd::Alignment) int8_t input[SpatialIn * 20 * 20];
  alignas(simd::Alignment) int16_t weights[SpatialIn * SpatialOut * 3 * 3];
  modInit(input, SpatialIn * 20 * 20, 13);
  modInit(weights, SpatialIn * SpatialOut * 3 * 3, 11);
  QConvNaive<SpatialIn, SpatialOut, 20, 3> q;
  q.initWeights(weights);
  for (auto _ : state) {
    q.propagate(input);
    benchmark::ClobberMemory();
  }
  checkTrue(q.outputBuf[0] != 1);
}
BENCHMARK(qconv_naive);

void qconv_simdops(benchmark::State& state)
{
  constexpr int SpatialIn = 16;
  constexpr int SpatialOut = 16;
  alignas(simd::Alignment) int8_t input[SpatialIn * 20 * 20];
  alignas(simd::Alignment) int16_t weights[SpatialIn * SpatialOut * 3 * 3];
  modInit(input, SpatialIn * 20 * 20, 13);
  modInit(weights, SpatialIn * SpatialOut * 3 * 3, 11);
  QConv<SpatialIn, SpatialOut, 20, 3> q;
  q.initWeights(weights);
  for (auto _ : state) {
    q.propagate(input);
    benchmark::ClobberMemory();
  }
  checkTrue(q.outputBuf[0] != 1);
}
BENCHMARK(qconv_simdops);

void qconv_thick_simdops(benchmark::State& state)
{
  constexpr int SpatialIn = 16;
  constexpr int SpatialOut = 16;
  alignas(simd::Alignment) int32_t input[SpatialIn * 20 * 20];
  alignas(simd::Alignment) int32_t inputPadded[SpatialIn * 22 * 22];
  alignas(simd::Alignment) int32_t unpaddedOutput[SpatialOut * 20 * 20];
  alignas(simd::Alignment) int32_t weights[SpatialIn * SpatialOut * 3 * 3];
  modInit(input, SpatialIn * 20 * 20, 13);
  modInit(weights, SpatialIn * SpatialOut * 3 * 3, 11);
  QConvThick<SpatialIn, SpatialOut, 20, 3> q;
  q.initWeights<int32_t>(weights);
  q.padInput(input, inputPadded);
  for (auto _ : state) {
    q.propagate(inputPadded);
    benchmark::ClobberMemory();
  }
  q.getUnpaddedOutput(unpaddedOutput);
  checkTrue(q.outputBuf[0] != 1);
}
BENCHMARK(qconv_thick_simdops);
BENCHMARK_MAIN();
