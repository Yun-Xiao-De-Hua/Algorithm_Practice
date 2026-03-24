#include<iostream>

typedef unsigned long long ull;
const int N = 1e3 + 10;
int T, n, m, q, u, v, x, y;
ull s[N][N], ans;

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);

	std::cin >> T;

	while (T--) {
		std::cin >> n >> m >> q;

		for (int i = 1; i <= n; i++)
			for (int j = 1; j <= m; j++) {
				std::cin >> s[i][j];
				s[i][j] += s[i - 1][j] + s[i][j - 1] - s[i - 1][j - 1];
			}

		while (q--) {
			std::cin >> u >> v >> x >> y;
			ans ^= s[x][y] - s[x][v - 1] - s[u - 1][y] + s[u - 1][v - 1];
		}

		std::cout << ans << '\n';
		ans = 0;
	}
	return 0;
}