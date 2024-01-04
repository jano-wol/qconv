// x86-64 gcc 11.4 -O3 -mavx2
#include <iostream>

constexpr size_t Size = 4096;
using T = int32_t;

T* add(T* out, T* in1, T* in2)
{
  for (int i = 0; i < Size; ++i) {
    out[i] = in1[i] + in2[i];
  }
  return out + Size;
}

int main()
{
  using T = int32_t;

  alignas(32) T a[Size];
  alignas(32) T b[Size];
  alignas(32) T c[Size];

  for (int i = 0; i < Size; ++i) {
    std::cin >> a[i];
    std::cin >> b[i];
  }

  add(c, a, b);
  for (int i = 0; i < Size; ++i) {
    std::cout << c[i];
  }
  std::cout << "\n";
}
