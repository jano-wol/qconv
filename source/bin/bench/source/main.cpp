#include <benchmark/benchmark.h>

#include <core/simdops.h>
#include <layers/add.h>

static void naive_zero_512_int8_t(benchmark::State& state)
{
  int8_t a[512];
  for (auto _ : state) {
    for (int i = 0; i < 512; ++i) {
      a[i] = 0;
    }
  }
}
BENCHMARK(naive_zero_512_int8_t);

static void simdops_zero_512_int8_t(benchmark::State& state)
{
  int8_t a[512];
  for (auto _ : state) {
    simdops::zero<512, int8_t>(a);
  }
}
BENCHMARK(simdops_zero_512_int8_t);

static void memset_zero_512_int8_t(benchmark::State& state)
{
  int8_t a[512];
  for (auto _ : state) {
    std::memset(a, 0, sizeof(int8_t) * 512);
  }
}
BENCHMARK(memset_zero_512_int8_t);

BENCHMARK_MAIN();
