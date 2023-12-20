#include <cstring>

#include <benchutils/benchutils.h>
#include <simdops/simdops.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::testutils;

template <int Size, typename T>
T minHNaive(T* input)
{
  T currMin = input[0];
  for (int i = 0; i < Size; ++i) {
    if (input[i] < currMin) {
      currMin = input[i];
    }
  }
  return currMin;
}

template <typename T, int Size>
void min_h_naive(benchmark::State& state)
{
  static_assert(std::is_signed<T>::value, "Test is implemented for signed types only!");
  alignas(simdops::NativeAlignment) T a[Size];
  modInit(a, Size, 8);
  a[59] = -7;
  T ret = 0;
  for (auto _ : state) {
    ret = minHNaive<Size, T>(a);
    benchmark::DoNotOptimize(ret);
  }
  std::ostream cnull(nullptr);
  cnull << ret;
  checkTrue(ret == -7);
}

template <typename T, int Size>
void min_h_simdops(benchmark::State& state)
{
  static_assert(std::is_signed<T>::value, "Test is implemented for signed types only!");
  alignas(simdops::NativeAlignment) T a[Size];
  modInit(a, Size, 8);
  a[59] = -7;
  T ret = 0;
  for (auto _ : state) {
    ret = simdops::minH<Size, T>(a);
    benchmark::DoNotOptimize(ret);
  }
  std::ostream cnull(nullptr);
  cnull << ret;
  checkTrue(ret == -7);
}
BENCH_SUITES_FOR_2(min_h_naive, min_h_simdops);

template <int Size, typename T>
T maxHNaive(T* input)
{
  T currMax = input[0];
  for (int i = 0; i < Size; ++i) {
    if (input[i] > currMax) {
      currMax = input[i];
    }
  }
  return currMax;
}

template <typename T, int Size>
void max_h_naive(benchmark::State& state)
{
  static_assert(std::is_signed<T>::value, "Test is implemented for signed types only!");
  alignas(simdops::NativeAlignment) T a[Size];
  modInit(a, Size, 8);
  a[59] = 71;
  T ret = 0;
  for (auto _ : state) {
    ret = maxHNaive<Size, T>(a);
    benchmark::DoNotOptimize(ret);
  }
  std::ostream cnull(nullptr);
  cnull << ret;
  checkTrue(ret == 71);
}

template <typename T, int Size>
void max_h_simdops(benchmark::State& state)
{
  static_assert(std::is_signed<T>::value, "Test is implemented for signed types only!");
  alignas(simdops::NativeAlignment) T a[Size];
  modInit(a, Size, 8);
  a[59] = 71;
  T ret = 0;
  for (auto _ : state) {
    ret = simdops::maxH<Size, T>(a);
    benchmark::DoNotOptimize(ret);
  }
  std::ostream cnull(nullptr);
  cnull << ret;
  checkTrue(ret == 71);
}
BENCH_SUITES_FOR_2(max_h_naive, max_h_simdops);

BENCHMARK_MAIN();
