#include <cstring>

#include <benchutils/benchutils.h>
#include <core/simdops.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::testutils;

template <typename T, int Size>
void zero_naive(benchmark::State& state)
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
void zero_simdops(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) T a[Size];
  modInit(a, Size, 11);
  for (auto _ : state) {
    simdops::zero<Size, T>(a);
  }
  checkTrue(a[1] == 0);
}

template <typename T, int Size>
static void zero_memset(benchmark::State& state)
{
  T a[Size];
  modInit(a, Size, 11);
  for (auto _ : state) {
    std::memset(a, 0, sizeof(T) * Size);
  }
  checkTrue(a[1] == 0);
}
BENCH_SUITES_FOR_3(zero_naive, zero_simdops, zero_memset);
BENCHMARK_MAIN();
