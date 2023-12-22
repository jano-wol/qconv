// x86-64 gcc 11.4 -O3 -mavx2
#include <iostream>

constexpr size_t Size = 4096;
using T = int32_t;

T* copyNaive(T* out, T* in)
{
  for (int i = 0; i < Size; ++i) {
    out[i] = in[i];
  }
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

  copyNaive(b, a);
  for (int i = 0; i < Size; ++i) {
    std::cout << b[i];
  }
  std::cout << "\n";
}
