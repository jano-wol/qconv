#include <core/simdops.h>
#include <layers/add.h>

int main()
{
  int8_t a[512];
  simdops::zero<512, int8_t>(a);
  return 0;
}
