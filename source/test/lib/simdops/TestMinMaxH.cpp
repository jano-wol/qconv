#include <gmock/gmock.h>

#include <simdops/simdops.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::simdops;
using namespace qconv::testutils;

TEST(MinMaxH, ConstInit)
{
  alignas(simdops::NativeAlignment) int8_t a[4096];
  constInit(a, 4096, int8_t(127));
  int8_t ret = simdops::minH<4096, int8_t>(a);
  EXPECT_EQ(ret, 127);
  a[1234] = 126;
  ret = simdops::minH<4096, int8_t>(a);
  EXPECT_EQ(ret, 126);

  constInit(a, 4096, int8_t(-128));
  ret = simdops::maxH<4096, int8_t>(a);
  EXPECT_EQ(ret, -128);
  a[2345] = -127;
  ret = simdops::maxH<4096, int8_t>(a);
  EXPECT_EQ(ret, -127);
}

TEST(MinMaxH, Mod8)
{
  alignas(simdops::NativeAlignment) int8_t a[64];
  modInit(a, 64, 8);
  a[59] = -7;
  int8_t ret = simdops::minH<64, int8_t>(a);
  EXPECT_EQ(ret, -7);

  ret = simdops::maxH<64, int8_t>(a);
  EXPECT_EQ(ret, 7);
}

TEST(MinMaxH, RandomInit)
{
  alignas(simdops::NativeAlignment) int32_t a[4096];
  randInit(a, 4096);
  int32_t ret = simdops::minH<64, int32_t>(a);
  EXPECT_EQ(ret, -2126938739);

  ret = simdops::maxH<64, int32_t>(a);
  EXPECT_EQ(ret, 2132285156);
}
