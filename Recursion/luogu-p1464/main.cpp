// 记忆化搜索，通过维护一个空间，记录已经计算过的数值，优化计算复杂度

#include<iostream>

typedef long long ll;
ll memo[25][25][25], a, b, c;

ll w(ll a, ll b, ll c)
{
	if (a <= 0 || b <= 0 || c <= 0) return 1;
	if (a > 20 || b > 20 || c > 20) return w(20, 20, 20);
	if (memo[a][b][c]) return memo[a][b][c];

	if (a < b && b < c) return memo[a][b][c] = w(a, b, c - 1) + w(a, b - 1, c - 1) - w(a, b - 1, c);
	else return memo[a][b][c] = w(a - 1, b, c) + w(a - 1, b - 1, c) + w(a - 1, b, c - 1) - w(a - 1, b - 1, c - 1);
}

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);

	while (std::cin >> a >> b >> c && (a != -1 || b != -1 || c != -1)) {
		std::cout << "w(" << a << ", " << b << ", " << c << ") = " << w(a, b, c) << '\n';
	}

	return 0;
}