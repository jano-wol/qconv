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

template <typename T>
void weightInit_32_512(T a[32][512])
{
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 512; ++j) {
      a[i][j] = 0;
    }
  }
  a[0][1] = 1;
  a[1][0] = 2;
  a[30][511] = 2;
  a[31][510] = 1;
  a[30][510] = 3;
  a[31][511] = 3;
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

// relu
static void naive_relu_512_int8_t(benchmark::State& state)
{
  int8_t a1[512];
  int8_t a2[512];
  int8_t b1[512];
  int8_t b2[512];
  constInit(a1, static_cast<int8_t>(4), 512);
  constInit(a2, static_cast<int8_t>(-1), 512);
  for (auto _ : state) {
    for (int i = 0; i < 512; ++i) {
      if (a1[i] >= 0) {
        b1[i] = a1[i];
      } else {
        b1[i] = 0;
      }
      if (a2[i] >= 0) {
        b2[i] = a2[i];
      } else {
        b2[i] = 0;
      }
    }
  }
  checkTrue(b1[3] == 4);
  checkTrue(b2[3] == 0);
}
BENCHMARK(naive_relu_512_int8_t);

static void simdops_relu_512_int8_t(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) int8_t a1[512];
  alignas(simdops::NativeAlignment) int8_t a2[512];
  alignas(simdops::NativeAlignment) int8_t b1[512];
  alignas(simdops::NativeAlignment) int8_t b2[512];
  constInit(a1, static_cast<int8_t>(4), 512);
  constInit(a2, static_cast<int8_t>(-1), 512);
  for (auto _ : state) {
    simdops::relu<512, int8_t>(b1, a1);
    simdops::relu<512, int8_t>(b2, a2);
  }
  checkTrue(b1[3] == 4);
  checkTrue(b2[3] == 0);
}
BENCHMARK(simdops_relu_512_int8_t);

// linear
static void naive_linear_512_int8_t(benchmark::State& state)
{
  int8_t in[512];
  int32_t bias[32];
  int8_t weight[32][512];
  int32_t out[32];

  mod8Init(in, 512);
  mod8Init(bias, 32);
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
  checkTrue(out[0] == 1);    // 1 * 1 + 0
  checkTrue(out[1] == 1);    // 2 * 0 + 1
  checkTrue(out[30] == 38);  // 3 * 6 + 2 * 7 + 6
  checkTrue(out[31] == 34);  // 1 * 6 + 3 * 7 + 7
}
BENCHMARK(naive_linear_512_int8_t);

#ifdef USE_AVX2
static void simdops_linear_512_int8_t(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) int8_t in[512];
  alignas(simdops::NativeAlignment) int32_t bias[32];
  alignas(simdops::NativeAlignment) int8_t weight[32][512];
  alignas(simdops::NativeAlignment) int32_t out[32];

  mod8Init(in, 512);
  mod8Init(bias, 32);
  weightInit_32_512(weight);
  for (auto _ : state) {
    simdops::linear<32, 512, 1>(out, in, weight, bias);
  }
  checkTrue(out[0] == 1);
  checkTrue(out[1] == 1);
  checkTrue(out[30] == 38);
  checkTrue(out[31] == 34);
}
BENCHMARK(simdops_linear_512_int8_t);
#endif

BENCHMARK_MAIN();
