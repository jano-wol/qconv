#include <cstring>

#include <benchutils/benchutils.h>
#include <simdops/simdops.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::testutils;

template <typename T, int Size>
void min_global_naive(benchmark::State& state)
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
void min_global_simdops(benchmark::State& state)
{
  static_assert(std::is_same<T, int8_t>::value, "Test is implemented only for int8_t!");
  alignas(simdops::NativeAlignment) T a[Size];
  modInit(a, Size, 8);
  a[59] = -7;
  T ret = 0;
  for (auto _ : state) {
    ret = simdops::minGlobal<Size>(a);
  }
  //checkTrue(ret == -7); TODO
}
BENCH_SUITES_FOR_1(min_global_naive);
BENCHMARK_TEMPLATE(min_global_simdops, int8_t, 64);
BENCHMARK_TEMPLATE(min_global_simdops, int8_t, 512);
BENCHMARK_TEMPLATE(min_global_simdops, int8_t, 4096);
#else
BENCH_SUITES_FOR_1(min_global_naive);
#endif

template <typename T, int Size>
void max_global_naive(benchmark::State& state)
{
  static_assert(std::is_signed<T>::value, "Test is implemented for signed types only!");
  T a[Size];
  modInit(a, Size, 8);
  a[59] = 71;
  T ret = 0;
  for (auto _ : state) {
    T currMax = -128;
    for (int i = 0; i < Size; ++i) {
      if (a[i] >= currMax) {
        currMax = a[i];
      }
    }
    ret = currMax;
  }
  // checkTrue(ret == -7);
}

#ifdef USE_AVX2
template <typename T, int Size>
void max_global_simdops(benchmark::State& state)
{
  static_assert(std::is_same<T, int8_t>::value, "Test is implemented only for int8_t!");
  alignas(simdops::NativeAlignment) T a[Size];
  modInit(a, Size, 8);
  a[59] = 71;
  T ret = 0;
  for (auto _ : state) {
    ret = simdops::maxGlobal<Size>(a);
  }
  // checkTrue(ret == 71); TODO
}
BENCH_SUITES_FOR_1(max_global_naive);
BENCHMARK_TEMPLATE(max_global_simdops, int8_t, 64);
BENCHMARK_TEMPLATE(max_global_simdops, int8_t, 512);
BENCHMARK_TEMPLATE(max_global_simdops, int8_t, 4096);
#else
BENCH_SUITES_FOR_1(max_global_naive);
#endif

BENCHMARK_MAIN();
