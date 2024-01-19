#ifndef QCONV_BENCHUTILS_BENCHUTILS_H_
#define QCONV_BENCHUTILS_BENCHUTILS_H_

#include <iostream>

#include <benchmark/benchmark.h>

void checkTrue(bool check)
{
  if (check == false) {
    std::cerr << "Unexpected result!\n";
    exit(1);
  }
}

#endif  // QCONV_BENCHUTILS_BENCHUTILS_H_
