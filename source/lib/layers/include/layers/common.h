#ifndef QCONV_LAYERS_COMMON_H_INCLUDED
#define QCONV_LAYERS_COMMON_H_INCLUDED

#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>

#if defined(USE_AVX2)
#include <immintrin.h>

#elif defined(USE_SSE41)
#include <smmintrin.h>

#elif defined(USE_SSSE3)
#include <tmmintrin.h>

#elif defined(USE_SSE2)
#include <emmintrin.h>

#elif defined(USE_MMX)
#include <mmintrin.h>

#elif defined(USE_NEON)
#include <arm_neon.h>
#endif

namespace qconv::layers
{
// Size of cache line (in bytes)
constexpr std::size_t CacheLineSize = 64;

// SIMD width (in bytes)
#if defined(USE_AVX2)
constexpr std::size_t SimdWidth = 32;

#elif defined(USE_SSE2)
constexpr std::size_t SimdWidth = 16;

#elif defined(USE_MMX)
constexpr std::size_t SimdWidth = 8;

#elif defined(USE_NEON)
constexpr std::size_t SimdWidth = 16;
#endif

constexpr std::size_t MaxSimdWidth = 32;

// Type of input feature after conversion
using IndexType = std::uint32_t;

}  // namespace qconv::layers

#endif  // #ifndef QCONV_LAYERS_COMMON_H_INCLUDED
