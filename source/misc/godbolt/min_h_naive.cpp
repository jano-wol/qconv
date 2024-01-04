// x86-64 gcc 11.4 -O3 -mavx2
#include <iostream>

constexpr size_t Size = 4096;
using T = int32_t;

T minH(T* a)
{
  T currMin = a[0];
  for (int i = 0; i < Size; ++i) {
    if (a[i] < currMin) {
      currMin = a[i];
    }
  }
  return currMin;
}

int main()
{
  using T = int32_t;

  alignas(32) T a[Size];

  for (int i = 0; i < Size; ++i) {
    std::cin >> a[i];
  }

  auto ret = minH(a);
  std::cout << ret << "\n";
}
