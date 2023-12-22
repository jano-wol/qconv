#include <cstring>

#include <benchutils/benchutils.h>
#include <simdops/simdops.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::testutils;

template <int Size, typename T>
T* addNaive(T* out, T* in1, T* in2)
{
  for (int i = 0; i < Size; ++i) {
    out[i] = in1[i] + in2[i];
  }
  return out + Size;
}

template <typename T, int Size>
void add_naive(benchmark::State& state)
{
  alignas(simdops::NativeAlignment) T a[Size];
  alignas(simdops::NativeAlignment) T b[Size];
  alignas(simdops::NativeAlignment) T c[Size];
  modInit(a, Size, 11);
  modInit(b, Size, 11);
  for (auto _ : state) {
    benchmark::DoNotOptimize(addNaive<Size, T>(c, a, b));
    benchmark::ClobberMemory();
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
    benchmark::DoNotOptimize(simdops::add<Size, T>(c, a, b));
    benchmark::ClobberMemory();
  }
  checkTrue(c[2] == 4);
}
BENCH_SUITES_FOR_2(add_naive, add_simdops);
BENCHMARK_MAIN();
