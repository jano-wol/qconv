#include <cstring>

#include <benchutils/benchutils.h>
#include <core/simdops.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::testutils;

template <typename T, int Size>
void add_naive(benchmark::State& state)
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
void add_simdops(benchmark::State& state)
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
BENCH_SUITES_FOR_2(add_naive, add_simdops);
BENCHMARK_MAIN();

