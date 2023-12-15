#include <core/utils.h>

#include <iostream>

using namespace qconv;

void core::printAs32s(simde__m256i v)
{
  int32_t values[8];
  simde_mm256_storeu_si256(values, v);
  for (int i = 0; i < 8; ++i) {
    std::cout << static_cast<int>(values[i]) << " ";
  }
  std::cout << "\n";
}

void core::printAs16s(simde__m256i v)
{
  int16_t values[16];
  simde_mm256_storeu_si256(values, v);
  for (int i = 0; i < 16; ++i) {
    std::cout << static_cast<int>(values[i]) << " ";
  }
  std::cout << "\n";
}

void core::printAs8s(simde__m256i v)
{
  int8_t values[32];
  simde_mm256_storeu_si256(values, v);
  for (int i = 0; i < 32; ++i) {
    std::cout << static_cast<int>(values[i]) << " ";
  }
  std::cout << "\n";
}
