#include <benchmark/benchmark.h>

#include <core/simdops.h>
#include <layers/add.h>

// utils
template <typename T>
void dummyInit(T* a, int s)
{
  for (int i = 0; i < s; ++i) {
    a[i] = i % 8;
  }
}

void checkTrue(bool check)
{
  if (check == false) {
    std::cerr << "Unexpected result!\n";
    exit(1);
  }
}

// zero
static void naive_zero_512_int8_t(benchmark::State& state)
{
  int8_t a[512];
  dummyInit(a, 512);
  for (auto _ : state) {
    for (int i = 0; i < 512; ++i) {
      a[i] = 0;
    }
  }
  checkTrue(a[1] == 0);
}
BENCHMARK(naive_zero_512_int8_t);

static void simdops_zero_512_int8_t(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) int8_t a[512];
  dummyInit(a, 512);
  for (auto _ : state) {
    simdops::zero<512, int8_t>(a);
  }
  checkTrue(a[1] == 0);
}
BENCHMARK(simdops_zero_512_int8_t);

static void memset_zero_512_int8_t(benchmark::State& state)
{
  int8_t a[512];
  dummyInit(a, 512);
  for (auto _ : state) {
    std::memset(a, 0, sizeof(int8_t) * 512);
  }
  checkTrue(a[1] == 0);
}
BENCHMARK(memset_zero_512_int8_t);

// copy
static void naive_copy_512_int8_t(benchmark::State& state)
{
  int8_t a[512];
  int8_t b[512];
  dummyInit(a, 512);
  for (auto _ : state) {
    for (int i = 0; i < 512; ++i) {
      b[i] = a[i];
    }
  }
  checkTrue(b[1] == 1);
}
BENCHMARK(naive_copy_512_int8_t);

static void simdops_copy_512_int8_t(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) int8_t a[512];
  alignas(simdops::NativeAlignment) int8_t b[512];
  dummyInit(a, 512);
  for (auto _ : state) {
    simdops::copy<512, int8_t>(b, a);
  }
  checkTrue(b[1] == 1);
}
BENCHMARK(simdops_copy_512_int8_t);

static void memcopy_copy_512_int8_t(benchmark::State& state)
{
  int8_t a[512];
  int8_t b[512];
  dummyInit(a, 512);
  for (auto _ : state) {
    std::memcpy(b, a, sizeof(int8_t) * 512);
  }
  checkTrue(b[1] == 1);
}
BENCHMARK(memcopy_copy_512_int8_t);

BENCHMARK_MAIN();
