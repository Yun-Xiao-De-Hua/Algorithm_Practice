#include<iostream>

int a, b, x, y;

int main()
{
    std::cin >> a >> b >> x >> y;

    int min = 20;

    for (int i = 1; i <= 20; ++i) {
        int cnt_a = std::max(0, (a - i * y + x - 1) / x);
        int cnt_b = std::max(0, (b - i * y + x - 1) / x);
        int cnt = cnt_a + cnt_b + i;
        if (cnt < min) min = cnt;
    }

    std::cout << min;

    return 0;
}