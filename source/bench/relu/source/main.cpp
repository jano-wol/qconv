#include <cstring>

#include <benchutils/benchutils.h>
#include <core/simdops.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::testutils;

template <typename T, int Size>
void relu_naive(benchmark::State& state)
{
  static_assert(std::is_signed<T>::value, "Test is implemented for signed types only!");
  T a1[Size];
  T a2[Size];
  T b1[Size];
  T b2[Size];
  constInit(a1, Size, static_cast<T>(4));
  constInit(a2, Size, static_cast<T>(-1));
  for (auto _ : state) {
    for (int i = 0; i < Size; ++i) {
      if (a1[i] >= 0) {
        b1[i] = a1[i];
      } else {
        b1[i] = 0;
      }
      if (a2[i] >= 0) {
        b2[i] = a2[i];
      } else {
        b2[i] = 0;
      }
    }
  }
  checkTrue(b1[3] == 4);
  checkTrue(b2[3] == 0);
}

template <typename T, int Size>
static void relu_simdops(benchmark::State& state)
{
  static_assert(std::is_signed<T>::value, "Test is implemented for signed types only!");
  alignas(simdops::NativeAlignment) T a1[Size];
  alignas(simdops::NativeAlignment) T a2[Size];
  alignas(simdops::NativeAlignment) T b1[Size];
  alignas(simdops::NativeAlignment) T b2[Size];
  constInit(a1, Size, static_cast<T>(4));
  constInit(a2, Size, static_cast<T>(-1));
  for (auto _ : state) {
    simdops::relu<Size, T>(b1, a1);
    simdops::relu<Size, T>(b2, a2);
  }
  checkTrue(b1[3] == 4);
  checkTrue(b2[3] == 0);
}
BENCH_SUITES_FOR_2(relu_naive, relu_simdops);
BENCHMARK_MAIN();
