#include <cstring>

#include <benchutils/benchutils.h>
#include <core/simdops.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::testutils;

template <typename T, int Size>
void min_naive(benchmark::State& state)
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
void min_simdops(benchmark::State& state)
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
BENCH_SUITES_FOR_2(min_naive, min_simdops);

template <typename T, int Size>
void max_naive(benchmark::State& state)
{
  T a[Size];
  T b[Size];
  T c[Size];
  constInit(a, Size, static_cast<T>(4));
  modInit(b, Size, 11);
  for (auto _ : state) {
    for (int i = 0; i < Size; ++i) {
      if (a[i] >= b[i]) {
        c[i] = a[i];
      } else {
        c[i] = b[i];
      }
    }
  }
  checkTrue(c[3] == 4);
}

template <typename T, int Size>
void max_simdops(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) T a[Size];
  alignas(simdops::NativeAlignment) T b[Size];
  alignas(simdops::NativeAlignment) T c[Size];
  constInit(a, Size, static_cast<T>(4));
  modInit(b, Size, 11);
  for (auto _ : state) {
    simdops::max<Size, T>(c, a, b);
  }
  checkTrue(c[3] == 4);
}
BENCH_SUITES_FOR_2(max_naive, max_simdops);
BENCHMARK_MAIN();
