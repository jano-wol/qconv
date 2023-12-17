#ifndef QCONV_BENCHUTILS_BENCHUTILS_H_
#define QCONV_BENCHUTILS_BENCHUTILS_H_

#include <iostream>

#include <benchmark/benchmark.h>

#define BENCH_SUITES_FOR_1(func1)           \
  BENCHMARK_TEMPLATE(func1, int8_t, 64);    \
  BENCHMARK_TEMPLATE(func1, int8_t, 512);   \
  BENCHMARK_TEMPLATE(func1, int8_t, 4096);  \
  BENCHMARK_TEMPLATE(func1, int16_t, 64);   \
  BENCHMARK_TEMPLATE(func1, int16_t, 512);  \
  BENCHMARK_TEMPLATE(func1, int16_t, 4096); \
  BENCHMARK_TEMPLATE(func1, int32_t, 64);   \
  BENCHMARK_TEMPLATE(func1, int32_t, 512);  \
  BENCHMARK_TEMPLATE(func1, int32_t, 4096);

#define BENCH_SUITES_FOR_2(func1, func2)    \
  BENCHMARK_TEMPLATE(func1, int8_t, 64);    \
  BENCHMARK_TEMPLATE(func2, int8_t, 64);    \
  BENCHMARK_TEMPLATE(func1, int8_t, 512);   \
  BENCHMARK_TEMPLATE(func2, int8_t, 512);   \
  BENCHMARK_TEMPLATE(func1, int8_t, 4096);  \
  BENCHMARK_TEMPLATE(func2, int8_t, 4096);  \
  BENCHMARK_TEMPLATE(func1, int16_t, 64);   \
  BENCHMARK_TEMPLATE(func2, int16_t, 64);   \
  BENCHMARK_TEMPLATE(func1, int16_t, 512);  \
  BENCHMARK_TEMPLATE(func2, int16_t, 512);  \
  BENCHMARK_TEMPLATE(func1, int16_t, 4096); \
  BENCHMARK_TEMPLATE(func2, int16_t, 4096); \
  BENCHMARK_TEMPLATE(func1, int32_t, 64);   \
  BENCHMARK_TEMPLATE(func2, int32_t, 64);   \
  BENCHMARK_TEMPLATE(func1, int32_t, 512);  \
  BENCHMARK_TEMPLATE(func2, int32_t, 512);  \
  BENCHMARK_TEMPLATE(func1, int32_t, 4096); \
  BENCHMARK_TEMPLATE(func2, int32_t, 4096);

#define BENCH_SUITES_FOR_3(func1, func2, func3) \
  BENCHMARK_TEMPLATE(func1, int8_t, 64);        \
  BENCHMARK_TEMPLATE(func2, int8_t, 64);        \
  BENCHMARK_TEMPLATE(func3, int8_t, 64);        \
  BENCHMARK_TEMPLATE(func1, int8_t, 512);       \
  BENCHMARK_TEMPLATE(func2, int8_t, 512);       \
  BENCHMARK_TEMPLATE(func3, int8_t, 512);       \
  BENCHMARK_TEMPLATE(func1, int8_t, 4096);      \
  BENCHMARK_TEMPLATE(func2, int8_t, 4096);      \
  BENCHMARK_TEMPLATE(func3, int8_t, 4096);      \
  BENCHMARK_TEMPLATE(func1, int16_t, 64);       \
  BENCHMARK_TEMPLATE(func2, int16_t, 64);       \
  BENCHMARK_TEMPLATE(func3, int16_t, 64);       \
  BENCHMARK_TEMPLATE(func1, int16_t, 512);      \
  BENCHMARK_TEMPLATE(func2, int16_t, 512);      \
  BENCHMARK_TEMPLATE(func3, int16_t, 512);      \
  BENCHMARK_TEMPLATE(func1, int16_t, 4096);     \
  BENCHMARK_TEMPLATE(func2, int16_t, 4096);     \
  BENCHMARK_TEMPLATE(func3, int16_t, 4096);     \
  BENCHMARK_TEMPLATE(func1, int32_t, 64);       \
  BENCHMARK_TEMPLATE(func2, int32_t, 64);       \
  BENCHMARK_TEMPLATE(func3, int32_t, 64);       \
  BENCHMARK_TEMPLATE(func1, int32_t, 512);      \
  BENCHMARK_TEMPLATE(func2, int32_t, 512);      \
  BENCHMARK_TEMPLATE(func3, int32_t, 512);      \
  BENCHMARK_TEMPLATE(func1, int32_t, 4096);     \
  BENCHMARK_TEMPLATE(func2, int32_t, 4096);     \
  BENCHMARK_TEMPLATE(func3, int32_t, 4096);

void checkTrue(bool check)
{
  if (check == false) {
    std::cerr << "Unexpected result!\n";
    exit(1);
  }
}

#endif  // QCONV_BENCHUTILS_BENCHUTILS_H_
