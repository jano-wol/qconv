#include <gmock/gmock.h>

#include <simdops/simdops.h>
#include <testutils/testutils.h>

using namespace qconv;
using namespace qconv::simdops;
using namespace qconv::testutils;

TEST(MinMaxGlobal, Test1)
{
  alignas(simdops::NativeAlignment) int8_t a[64];
  modInit(a, 64, 8);
  a[59] = -7;
  int8_t ret = simdops::minGlobal<64, int8_t>(a);
  EXPECT_EQ(ret, -7);
}
