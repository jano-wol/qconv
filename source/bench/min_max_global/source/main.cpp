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

template <typename T, int Size>
void min_global_simdops(benchmark::State& state)
{
  static_assert(std::is_signed<T>::value, "Test is implemented for signed types only!");
  alignas(simdops::NativeAlignment) T a[Size];
  modInit(a, Size, 100);
  a[59] = -7;
  T ret = 0;
  for (auto _ : state) {
    ret = simdops::minGlobal<Size, T>(a);
  }
  checkTrue(ret == -7);
}
BENCH_SUITES_FOR_2(min_global_naive, min_global_simdops);

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
  checkTrue(ret == 71);
}

template <typename T, int Size>
void max_global_simdops(benchmark::State& state)
{
  static_assert(std::is_signed<T>::value, "Test is implemented for signed types only!");
  alignas(simdops::NativeAlignment) T a[Size];
  modInit(a, Size, 8);
  a[59] = 71;
  T ret = 0;
  for (auto _ : state) {
    ret = simdops::maxGlobal<Size, T>(a);
  }
  checkTrue(ret == 71);
}
BENCH_SUITES_FOR_2(max_global_naive, max_global_simdops);

BENCHMARK_MAIN();
