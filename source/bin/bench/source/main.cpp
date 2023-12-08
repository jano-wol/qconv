#include <benchmark/benchmark.h>

#include <core/simdops.h>
#include <layers/add.h>

// utils
template <typename T>
void constInit(T* a, T v, int s)
{
  for (int i = 0; i < s; ++i) {
    a[i] = v;
  }
}

template <typename T>
void mod8Init(T* a, int s)
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
  mod8Init(a, 512);
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
  mod8Init(a, 512);
  for (auto _ : state) {
    simdops::zero<512, int8_t>(a);
  }
  checkTrue(a[1] == 0);
}
BENCHMARK(simdops_zero_512_int8_t);

static void memset_zero_512_int8_t(benchmark::State& state)
{
  int8_t a[512];
  mod8Init(a, 512);
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
  mod8Init(a, 512);
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
  mod8Init(a, 512);
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
  mod8Init(a, 512);
  for (auto _ : state) {
    std::memcpy(b, a, sizeof(int8_t) * 512);
  }
  checkTrue(b[1] == 1);
}
BENCHMARK(memcopy_copy_512_int8_t);

// dilate
static void naive_dilate_512_int8_t(benchmark::State& state)
{
  int8_t a[512];
  int8_t b[512];
  mod8Init(a, 512);
  for (auto _ : state) {
    for (int i = 0; i < 512; ++i) {
      b[i] = a[i] + 1;
    }
  }
  checkTrue(b[1] == 2);
}
BENCHMARK(naive_dilate_512_int8_t);

static void simdops_dilate_512_int8_t(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) int8_t a[512];
  alignas(simdops::NativeAlignment) int8_t b[512];
  mod8Init(a, 512);
  for (auto _ : state) {
    simdops::add<512, int8_t>(b, a, 1);
  }
  checkTrue(b[1] == 2);
}
BENCHMARK(simdops_dilate_512_int8_t);

// add
static void naive_add_512_int8_t(benchmark::State& state)
{
  int8_t a[512];
  int8_t b[512];
  int8_t c[512];
  mod8Init(a, 512);
  mod8Init(b, 512);
  for (auto _ : state) {
    for (int i = 0; i < 512; ++i) {
      c[i] = a[i] + b[i];
    }
  }
  checkTrue(c[2] == 4);
}
BENCHMARK(naive_add_512_int8_t);

static void simdops_add_512_int8_t(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) int8_t a[512];
  alignas(simdops::NativeAlignment) int8_t b[512];
  alignas(simdops::NativeAlignment) int8_t c[512];
  mod8Init(a, 512);
  mod8Init(b, 512);
  for (auto _ : state) {
    simdops::add<512, int8_t>(c, a, b);
  }
  checkTrue(c[2] == 4);
}
BENCHMARK(simdops_add_512_int8_t);

// min
static void naive_min_512_int8_t(benchmark::State& state)
{
  int8_t a[512];
  int8_t b[512];
  int8_t c[512];
  constInit(a, static_cast<int8_t>(4), 512);
  mod8Init(b, 512);
  for (auto _ : state) {
    for (int i = 0; i < 512; ++i) {
      if (a[i] <= b[i]) {
        c[i] = a[i];
      } else {
        c[i] = b[i];
      }
    }
  }
  checkTrue(c[5] == 4);
}
BENCHMARK(naive_min_512_int8_t);

static void simdops_min_512_int8_t(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) int8_t a[512];
  alignas(simdops::NativeAlignment) int8_t b[512];
  alignas(simdops::NativeAlignment) int8_t c[512];
  constInit(a, static_cast<int8_t>(4), 512);
  mod8Init(b, 512);
  for (auto _ : state) {
    simdops::min<512, int8_t>(c, a, b);
  }
  checkTrue(c[5] == 4);
}
BENCHMARK(simdops_min_512_int8_t);

// max
static void naive_max_512_int8_t(benchmark::State& state)
{
  int8_t a[512];
  int8_t b[512];
  int8_t c[512];
  constInit(a, static_cast<int8_t>(4), 512);
  mod8Init(b, 512);
  for (auto _ : state) {
    for (int i = 0; i < 512; ++i) {
      if (a[i] >= b[i]) {
        c[i] = a[i];
      } else {
        c[i] = b[i];
      }
    }
  }
  checkTrue(c[3] == 4);
}
BENCHMARK(naive_max_512_int8_t);

static void simdops_max_512_int8_t(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) int8_t a[512];
  alignas(simdops::NativeAlignment) int8_t b[512];
  alignas(simdops::NativeAlignment) int8_t c[512];
  constInit(a, static_cast<int8_t>(4), 512);
  mod8Init(b, 512);
  for (auto _ : state) {
    simdops::max<512, int8_t>(c, a, b);
  }
  checkTrue(c[3] == 4);
}
BENCHMARK(simdops_max_512_int8_t);

BENCHMARK_MAIN();
