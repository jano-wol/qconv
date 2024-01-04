#ifndef QCONV_SIMD_SIMD_H_INCLUDED
#define QCONV_SIMD_SIMD_H_INCLUDED

#include <cstdint>
#include <cstdio>

#if defined(USE_AVX2)
#include <immintrin.h>
#else
static_assert(false, "Currently only AVX2 is supported");
#endif

namespace qconv::simd
{
#if defined(USE_AVX2)
constexpr size_t Alignment = 32;
constexpr size_t RegisterWidth = 256;
#else
static_assert(false, "Currently only AVX2 is supported");
#endif

template <typename T>
constexpr bool isPtrAligned(const T* pointer)
{
  return (reinterpret_cast<uintptr_t>(pointer) & (Alignment - 1)) == 0;
}
}  // namespace qconv::simd

#endif  // #ifndef QCONV_SIMD_SIMD_H_INCLUDED
