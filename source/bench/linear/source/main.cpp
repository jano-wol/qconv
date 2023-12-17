#include <cstring>

#include <benchutils/benchutils.h>
#include <core/simdops.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::testutils;

void linear_naive_512x32_int8_t(benchmark::State& state)
{
  int8_t in[512];
  int32_t bias[32];
  int8_t weight[32][512];
  int32_t out[32];

  modInit(in, 512, 11);
  modInit(bias, 32, 11);
  weightInit_32_512(weight);
  for (auto _ : state) {
    for (int i = 0; i < 32; ++i) {
      int sum = bias[i];
      for (int j = 0; j < 512; ++j) {
        sum += in[j] * weight[i][j];
      }
      out[i] = sum;
    }
  }
  checkTrue(out[0] == 161802);
  checkTrue(out[1] == 161532);
  checkTrue(out[30] == 162384);
  checkTrue(out[31] == 161986);
}
BENCHMARK(linear_naive_512x32_int8_t);

#ifdef USE_AVX2
void linear_simdops_512x32_int8_t(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) int8_t in[512];
  alignas(simdops::NativeAlignment) int32_t bias[32];
  alignas(simdops::NativeAlignment) int8_t weight[32][512];
  alignas(simdops::NativeAlignment) int32_t out[32];
  modInit(in, 512, 11);
  modInit(bias, 32, 11);
  weightInit_32_512(weight);
  for (auto _ : state) {
    simdops::linear<32, 512, 1>(out, in, weight, bias);
  }
  checkTrue(out[0] == 161802);
  checkTrue(out[1] == 161532);
  checkTrue(out[30] == 162384);
  checkTrue(out[31] == 161986);
}
BENCHMARK(linear_simdops_512x32_int8_t);
#endif

void linear_naive_512x32_float(benchmark::State& state)
{
  float in[512];
  float bias[32];
  float weight[32][512];
  float out[32];
  for (int i = 0; i < 512; ++i) {
    in[i] = 1.0 / static_cast<float>(i + 1);
  }
  for (int i = 0; i < 32; ++i) {
    bias[i] = 1.0 / static_cast<float>(i + 1);
  }
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 512; ++j) {
      weight[i][j] = 1.0 / static_cast<float>(i + j + 1);
    }
  }
  for (auto _ : state) {
    for (int i = 0; i < 32; ++i) {
      float sum = bias[i];
      for (int j = 0; j < 512; ++j) {
        sum += in[j] * weight[i][j];
      }
      out[i] = sum;
    }
  }
  checkTrue(2.64 < out[0] && out[0] < 2.65);
  checkTrue(1.49 < out[1] && out[1] < 1.50);
  checkTrue(1.08 < out[2] && out[2] < 1.09);
  checkTrue(0.85 < out[3] && out[3] < 0.86);
}
BENCHMARK(linear_naive_512x32_float);

void linear_simdops_512x32_float(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) float in[512];
  alignas(simdops::NativeAlignment) float bias[32];
  alignas(simdops::NativeAlignment) float weight[512][32];
  alignas(simdops::NativeAlignment) float out[32];
  for (int i = 0; i < 512; ++i) {
    in[i] = 1.0 / static_cast<float>(i + 1);
  }
  for (int i = 0; i < 32; ++i) {
    bias[i] = 1.0 / static_cast<float>(i + 1);
  }
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 512; ++j) {
      weight[j][i] = 1.0 / static_cast<float>(i + j + 1);
    }
  }
  for (auto _ : state) {
    simdops::linearLayer<simdops::Activation::None, 32, 512, float>(out, in, weight, bias);
  }
  checkTrue(2.64 < out[0] && out[0] < 2.65);
  checkTrue(1.49 < out[1] && out[1] < 1.50);
  checkTrue(1.08 < out[2] && out[2] < 1.09);
  checkTrue(0.85 < out[3] && out[3] < 0.86);
}
BENCHMARK(linear_simdops_512x32_float);
BENCHMARK_MAIN();
