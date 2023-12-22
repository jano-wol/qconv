// x86-64 gcc 11.4 -O3 -mavx2
#include <iostream>

constexpr size_t Size = 4096;
using T = int32_t;

T* zeroNaive(T* out)
{
  for (int i = 0; i < Size; ++i) {
    out[i] = 0;
  }
  return out + Size;
}

int main()
{
  using T = int32_t;

  alignas(32) T a[Size];

  for (int i = 0; i < Size; ++i) {
    std::cin >> a[i];
  }

  zeroNaive(a);
  for (int i = 0; i < Size; ++i) {
    std::cout << a[i];
  }
  std::cout << "\n";
}
