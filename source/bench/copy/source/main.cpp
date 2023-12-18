#include <cstring>

#include <benchutils/benchutils.h>
#include <simdops/simdops.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::testutils;

template <typename T, int Size>
void copy_naive(benchmark::State& state)
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
void copy_simdops(benchmark::State& state)
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
void copy_memcopy(benchmark::State& state)
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
BENCH_SUITES_FOR_3(copy_naive, copy_simdops, copy_memcopy);
BENCHMARK_MAIN();

