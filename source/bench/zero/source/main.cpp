#include <cstring>

#include <benchutils/benchutils.h>
#include <simdops/simdops.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::testutils;

template <typename T, int Size>
T* zeroNaive(T* out)
{
  for (int i = 0; i < Size; ++i) {
    out[i] = 0;
  }
  return out + Size;
}

template <typename T, int Size>
void zero_naive(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) T a[Size];
  modInit(a, Size, 11);
  for (auto _ : state) {
    benchmark::DoNotOptimize(zeroNaive<T, Size>(a));
    benchmark::ClobberMemory();
  }
  checkTrue(a[1] == 0);
}

template <typename T, int Size>
void zero_simdops(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) T a[Size];
  modInit(a, Size, 11);
  for (auto _ : state) {
    benchmark::DoNotOptimize(simdops::zero<Size, T>(a));
    benchmark::ClobberMemory();
  }
  checkTrue(a[1] == 0);
}

template <typename T, int Size>
void zero_memset(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) T a[Size];
  modInit(a, Size, 11);
  for (auto _ : state) {
    benchmark::DoNotOptimize(std::memset(a, 0, sizeof(T) * Size));
    benchmark::ClobberMemory();
  }
  checkTrue(a[1] == 0);
}
BENCH_SUITES_FOR_3(zero_naive, zero_simdops, zero_memset);
BENCHMARK_MAIN();
