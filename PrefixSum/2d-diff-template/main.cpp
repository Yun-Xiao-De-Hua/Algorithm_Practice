#include<iostream>

const int N = 1e3 + 10;
using ll = long long;
ll a[N][N], d[N][N];

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	int n, m, q; std::cin >> n >> m >> q;

	for (int i = 1; i <= n; i++)
		for (int j = 1; j <= m; j++)
			std::cin >> a[i][j];

	for(int i=1; i<=n; i++)
		for (int j = 1; j <= m; j++) {
			d[i][j] += a[i][j];
			d[i + 1][j] -= a[i][j];
			d[i][j + 1] -= a[i][j];
			d[i + 1][j + 1] += a[i][j];
		}

	while (q--) {
		int x1, y1, x2, y2, c; std::cin >> x1 >> y1 >> x2 >> y2 >> c;

		d[x1][y1] += c;
		d[x2 + 1][y1] -= c;
		d[x1][y2 + 1] -= c;
		d[x2 + 1][y2 + 1] += c;
	}

	for (int i = 1; i <= n; i++)
		for (int j = 1; j <= m; j++)
			a[i][j] = a[i - 1][j] + a[i][j - 1] - a[i - 1][j - 1] + d[i][j];

	for (int i = 1; i <= n; i++) {
		for (int j = 1; j <= m; j++) {
			std::cout << a[i][j] << (j != m ? " " : "");
		}
		std::cout << '\n';
	}

	return 0;
}