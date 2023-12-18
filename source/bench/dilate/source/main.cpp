#include <cstring>

#include <benchutils/benchutils.h>
#include <simdops/simdops.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::testutils;

// dilate
template <typename T, int Size>
void dilate_naive(benchmark::State& state)
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
void dilate_simdops(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) T a[Size];
  alignas(simdops::NativeAlignment) T b[Size];
  modInit(a, Size, 11);
  for (auto _ : state) {
    simdops::add<Size, T>(b, a, 1);
  }
  checkTrue(b[1] == 2);
}
BENCH_SUITES_FOR_2(dilate_naive, dilate_simdops);
BENCHMARK_MAIN();

