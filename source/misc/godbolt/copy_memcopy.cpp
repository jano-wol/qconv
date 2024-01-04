// x86-64 gcc 11.4 -O3 -mavx2
#include <cstring>
#include <iostream>

constexpr uint32_t Size = 4096;
using T = int32_t;

T* copyMemcopy(T* out, T* in)
{
  std::memcpy(out, in, sizeof(T) * Size);
  return out + Size;
}

int main()
{
  using T = int32_t;

  alignas(32) T a[Size];
  alignas(32) T b[Size];

  for (int i = 0; i < Size; ++i) {
    std::cin >> a[i];
  }

  copyMemcopy(b, a);
  for (int i = 0; i < Size; ++i) {
    std::cout << b[i];
  }
  std::cout << "\n";
}
