#include<iostream>

const int N = 1e3 + 10;
using ll = long long;
ll a[N][N], p[N][N];

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	int n, m, q; std::cin >> n >> m >> q;

	for (int i = 1; i <= n; i++)
		for (int j = 1; j <= m; j++)
			std::cin >> a[i][j];

	for (int i = 1; i <= n; i++)
		for (int j = 1; j <= m; j++)
			p[i][j] = p[i - 1][j] + p[i][j - 1] - p[i - 1][j - 1] + a[i][j];

	while (q--) {
		int x1, y1, x2, y2; std::cin >> x1 >> y1 >> x2 >> y2;
		std::cout << p[x2][y2] - p[x2][y1 - 1] - p[x1 - 1][y2] + p[x1 - 1][y1 - 1] << '\n';
	}

	return 0;
}