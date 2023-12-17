#include <benchmark/benchmark.h>

#include <core/simdops.h>
#include <core/utils.h>
#include <layers/qconv.h>
#include <layers/qconv_naive.h>

#define BENCH_SUITES_FOR_1(func1)           \
  BENCHMARK_TEMPLATE(func1, int8_t, 64);    \
  BENCHMARK_TEMPLATE(func1, int8_t, 512);   \
  BENCHMARK_TEMPLATE(func1, int8_t, 4096);  \
  BENCHMARK_TEMPLATE(func1, int16_t, 64);   \
  BENCHMARK_TEMPLATE(func1, int16_t, 512);  \
  BENCHMARK_TEMPLATE(func1, int16_t, 4096); \
  BENCHMARK_TEMPLATE(func1, int32_t, 64);   \
  BENCHMARK_TEMPLATE(func1, int32_t, 512);  \
  BENCHMARK_TEMPLATE(func1, int32_t, 4096);

#define BENCH_SUITES_FOR_2(func1, func2)    \
  BENCHMARK_TEMPLATE(func1, int8_t, 64);    \
  BENCHMARK_TEMPLATE(func2, int8_t, 64);    \
  BENCHMARK_TEMPLATE(func1, int8_t, 512);   \
  BENCHMARK_TEMPLATE(func2, int8_t, 512);   \
  BENCHMARK_TEMPLATE(func1, int8_t, 4096);  \
  BENCHMARK_TEMPLATE(func2, int8_t, 4096);  \
  BENCHMARK_TEMPLATE(func1, int16_t, 64);   \
  BENCHMARK_TEMPLATE(func2, int16_t, 64);   \
  BENCHMARK_TEMPLATE(func1, int16_t, 512);  \
  BENCHMARK_TEMPLATE(func2, int16_t, 512);  \
  BENCHMARK_TEMPLATE(func1, int16_t, 4096); \
  BENCHMARK_TEMPLATE(func2, int16_t, 4096); \
  BENCHMARK_TEMPLATE(func1, int32_t, 64);   \
  BENCHMARK_TEMPLATE(func2, int32_t, 64);   \
  BENCHMARK_TEMPLATE(func1, int32_t, 512);  \
  BENCHMARK_TEMPLATE(func2, int32_t, 512);  \
  BENCHMARK_TEMPLATE(func1, int32_t, 4096); \
  BENCHMARK_TEMPLATE(func2, int32_t, 4096);

#define BENCH_SUITES_FOR_3(func1, func2, func3) \
  BENCHMARK_TEMPLATE(func1, int8_t, 64);        \
  BENCHMARK_TEMPLATE(func2, int8_t, 64);        \
  BENCHMARK_TEMPLATE(func3, int8_t, 64);        \
  BENCHMARK_TEMPLATE(func1, int8_t, 512);       \
  BENCHMARK_TEMPLATE(func2, int8_t, 512);       \
  BENCHMARK_TEMPLATE(func3, int8_t, 512);       \
  BENCHMARK_TEMPLATE(func1, int8_t, 4096);      \
  BENCHMARK_TEMPLATE(func2, int8_t, 4096);      \
  BENCHMARK_TEMPLATE(func3, int8_t, 4096);      \
  BENCHMARK_TEMPLATE(func1, int16_t, 64);       \
  BENCHMARK_TEMPLATE(func2, int16_t, 64);       \
  BENCHMARK_TEMPLATE(func3, int16_t, 64);       \
  BENCHMARK_TEMPLATE(func1, int16_t, 512);      \
  BENCHMARK_TEMPLATE(func2, int16_t, 512);      \
  BENCHMARK_TEMPLATE(func3, int16_t, 512);      \
  BENCHMARK_TEMPLATE(func1, int16_t, 4096);     \
  BENCHMARK_TEMPLATE(func2, int16_t, 4096);     \
  BENCHMARK_TEMPLATE(func3, int16_t, 4096);     \
  BENCHMARK_TEMPLATE(func1, int32_t, 64);       \
  BENCHMARK_TEMPLATE(func2, int32_t, 64);       \
  BENCHMARK_TEMPLATE(func3, int32_t, 64);       \
  BENCHMARK_TEMPLATE(func1, int32_t, 512);      \
  BENCHMARK_TEMPLATE(func2, int32_t, 512);      \
  BENCHMARK_TEMPLATE(func3, int32_t, 512);      \
  BENCHMARK_TEMPLATE(func1, int32_t, 4096);     \
  BENCHMARK_TEMPLATE(func2, int32_t, 4096);     \
  BENCHMARK_TEMPLATE(func3, int32_t, 4096);

using namespace qconv;
using namespace qconv::core;
using namespace qconv::layers;

// utils
void checkTrue(bool check)
{
  if (check == false) {
    std::cerr << "Unexpected result!\n";
    exit(1);
  }
}

// zero
template <typename T, int Size>
void naive_zero(benchmark::State& state)
{
  T a[Size];
  modInit(a, Size, 11);
  for (auto _ : state) {
    for (int i = 0; i < Size; ++i) {
      a[i] = 0;
    }
  }
  checkTrue(a[1] == 0);
}

template <typename T, int Size>
void simdops_zero(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) T a[Size];
  modInit(a, Size, 11);
  for (auto _ : state) {
    simdops::zero<Size, T>(a);
  }
  checkTrue(a[1] == 0);
}

template <typename T, int Size>
static void memset_zero(benchmark::State& state)
{
  T a[Size];
  modInit(a, Size, 11);
  for (auto _ : state) {
    std::memset(a, 0, sizeof(T) * Size);
  }
  checkTrue(a[1] == 0);
}
BENCH_SUITES_FOR_3(naive_zero, simdops_zero, memset_zero);

// copy
template <typename T, int Size>
void naive_copy(benchmark::State& state)
{
  T a[Size];
  T b[Size];
  modInit(a, Size, 11);
  for (auto _ : state) {
    for (int i = 0; i < Size; ++i) {
      b[i] = a[i];
    }
  }
  checkTrue(b[1] == 1);
}

template <typename T, int Size>
void simdops_copy(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) T a[Size];
  alignas(simdops::NativeAlignment) T b[Size];
  modInit(a, Size, 11);
  for (auto _ : state) {
    simdops::copy<Size, T>(b, a);
  }
  checkTrue(b[1] == 1);
}

template <typename T, int Size>
void memcopy_copy(benchmark::State& state)
{
  T a[Size];
  T b[Size];
  std::memset(a, 0, sizeof(T) * Size);
  std::memset(b, 0, sizeof(T) * Size);
  modInit(a, Size, 11);
  for (auto _ : state) {
    std::memcpy(b, a, sizeof(T) * Size);
  }
  checkTrue(b[1] == 1);
}
BENCH_SUITES_FOR_3(naive_copy, simdops_copy, memcopy_copy);

// dilate
template <typename T, int Size>
void naive_dilate(benchmark::State& state)
{
  T a[Size];
  T b[Size];
  modInit(a, Size, 11);
  for (auto _ : state) {
    for (int i = 0; i < Size; ++i) {
      b[i] = a[i] + 1;
    }
  }
  checkTrue(b[1] == 2);
}

template <typename T, int Size>
void simdops_dilate(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) T a[Size];
  alignas(simdops::NativeAlignment) T b[Size];
  modInit(a, Size, 11);
  for (auto _ : state) {
    simdops::add<Size, T>(b, a, 1);
  }
  checkTrue(b[1] == 2);
}
BENCH_SUITES_FOR_2(naive_dilate, simdops_dilate);

// add
template <typename T, int Size>
void naive_add(benchmark::State& state)
{
  T a[Size];
  T b[Size];
  T c[Size];
  modInit(a, Size, 11);
  modInit(b, Size, 11);
  for (auto _ : state) {
    for (int i = 0; i < Size; ++i) {
      c[i] = a[i] + b[i];
    }
  }
  checkTrue(c[2] == 4);
}

template <typename T, int Size>
void simdops_add(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) T a[Size];
  alignas(simdops::NativeAlignment) T b[Size];
  alignas(simdops::NativeAlignment) T c[Size];
  modInit(a, Size, 11);
  modInit(b, Size, 11);
  for (auto _ : state) {
    simdops::add<Size, T>(c, a, b);
  }
  checkTrue(c[2] == 4);
}
BENCH_SUITES_FOR_2(naive_add, simdops_add);

// min
template <typename T, int Size>
void naive_min(benchmark::State& state)
{
  T a[Size];
  T b[Size];
  T c[Size];
  constInit(a, Size, static_cast<T>(4));
  modInit(b, Size, 11);
  for (auto _ : state) {
    for (int i = 0; i < Size; ++i) {
      if (a[i] <= b[i]) {
        c[i] = a[i];
      } else {
        c[i] = b[i];
      }
    }
  }
  checkTrue(c[5] == 4);
}

template <typename T, int Size>
void simdops_min(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) T a[Size];
  alignas(simdops::NativeAlignment) T b[Size];
  alignas(simdops::NativeAlignment) T c[Size];
  constInit(a, Size, static_cast<T>(4));
  modInit(b, Size, 11);
  for (auto _ : state) {
    simdops::min<Size, T>(c, a, b);
  }
  checkTrue(c[5] == 4);
}
BENCH_SUITES_FOR_2(naive_min, simdops_min);

// minGlobal
template <typename T, int Size>
void naive_min_global(benchmark::State& state)
{
  static_assert(std::is_signed<T>::value, "Test is implemented for signed types only!");
  T a[Size];
  modInit(a, Size, 8);
  a[59] = -7;
  T ret = 0;
  for (auto _ : state) {
    T currMin = 127;
    for (int i = 0; i < Size; ++i) {
      if (a[i] <= currMin) {
        currMin = a[i];
      }
    }
    ret = currMin;
  }
  checkTrue(ret == -7);
}

#ifdef USE_AVX2
template <typename T, int Size>
void simdops_min_global(benchmark::State& state)
{
  static_assert(std::is_same<T, int8_t>::value, "Test is currently implemented only for int8_t!");
  alignas(simdops::NativeAlignment) T a[Size];
  modInit(a, Size, 8);
  a[59] = -7;
  T ret = 0;
  for (auto _ : state) {
    ret = simdops::minGlobal<Size>(a);
  }
  std::cerr << "ret=" << int(ret) << "\n";
  checkTrue(ret == -7);
}
BENCH_SUITES_FOR_1(naive_min_global);
BENCHMARK_TEMPLATE(simdops_min_global, int8_t, 512);
BENCHMARK_TEMPLATE(simdops_min_global, int8_t, 512);
BENCHMARK_TEMPLATE(simdops_min_global, int8_t, 4096);
#else
BENCH_SUITES_FOR_1(naive_min_global);
#endif

// max
static void naive_max_512_int8_t(benchmark::State& state)
{
  int8_t a[512];
  int8_t b[512];
  int8_t c[512];
  constInit(a, 512, static_cast<int8_t>(4));
  modInit(b, 512, 11);
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
  constInit(a, 512, static_cast<int8_t>(4));
  modInit(b, 512, 11);
  for (auto _ : state) {
    simdops::max<512, int8_t>(c, a, b);
  }
  checkTrue(c[3] == 4);
}
BENCHMARK(simdops_max_512_int8_t);

// maxGlobal
static void naive_max_global_512_int8_t(benchmark::State& state)
{
  int8_t a[512];
  modInit(a, 512, 8);
  a[321] = 71;
  int8_t ret = 0;
  for (auto _ : state) {
    int8_t currMax = -128;
    for (int i = 0; i < 512; ++i) {
      if (a[i] >= currMax) {
        currMax = a[i];
      }
    }
    ret = currMax;
  }
  checkTrue(ret == 71);
}
BENCHMARK(naive_max_global_512_int8_t);

#ifdef USE_AVX2
static void simdops_max_global_512_int8_t(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) int8_t a[512];
  modInit(a, 512, 8);
  a[321] = 71;
  int8_t ret = 0;
  for (auto _ : state) {
    ret = simdops::maxGlobal<512>(a);
  }
  checkTrue(ret == 71);
}
BENCHMARK(simdops_max_global_512_int8_t);
#endif

// relu
static void naive_relu_512_int8_t(benchmark::State& state)
{
  int8_t a1[512];
  int8_t a2[512];
  int8_t b1[512];
  int8_t b2[512];
  constInit(a1, 512, static_cast<int8_t>(4));
  constInit(a2, 512, static_cast<int8_t>(-1));
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
  constInit(a1, 512, static_cast<int8_t>(4));
  constInit(a2, 512, static_cast<int8_t>(-1));
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
BENCHMARK(naive_linear_512_int8_t);

#ifdef USE_AVX2
static void simdops_linear_512_int8_t(benchmark::State& state)
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
BENCHMARK(simdops_linear_512_int8_t);
#endif

static void naive_linear_512_float(benchmark::State& state)
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
BENCHMARK(naive_linear_512_float);

static void simdops_linear_512_float(benchmark::State& state)
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
BENCHMARK(simdops_linear_512_float);

// #ifdef USE_AVX2
static void simdops_qconv(benchmark::State& state)
{
  constexpr int SpatialIn = 16;
  constexpr int SpatialOut = 16;
  alignas(simdops::NativeAlignment) int8_t input[SpatialIn * SpatialOut * 20 * 20];
  alignas(simdops::NativeAlignment) int16_t weights[SpatialIn * SpatialOut * 3 * 3];
  modInit(input, SpatialIn * SpatialOut * 20 * 20, 13);
  modInit(weights, SpatialIn * SpatialOut * 3 * 3, 11);
  QConv<SpatialIn, SpatialOut, 20, 3> q;
  q.initWeights(weights);

  for (auto _ : state) {
    q.propagate(input);
  }
  checkTrue(1 == 1);
}
BENCHMARK(simdops_qconv);

static void simdops_qconv_naive(benchmark::State& state)
{
  constexpr int SpatialIn = 16;
  constexpr int SpatialOut = 16;
  alignas(simdops::NativeAlignment) int8_t input[SpatialIn * SpatialOut * 20 * 20];
  alignas(simdops::NativeAlignment) int16_t weights[SpatialIn * SpatialOut * 3 * 3];
  modInit(input, SpatialIn * 20 * 20, 13);
  modInit(weights, SpatialIn * SpatialOut * 3 * 3, 11);
  QConvNaive<SpatialIn, SpatialOut, 20, 3> q;
  q.initWeights(weights);

  for (auto _ : state) {
    q.propagate(input);
  }
  checkTrue(q.outputBuf[0] != 1);
}
BENCHMARK(simdops_qconv_naive);
// #endif

BENCHMARK_MAIN();
