#include<iostream>

int n, a[35][35];
int x[5] = { 0,-1,1,0,0 };
int y[5] = { 0,0,0,-1,1 };

void dfs(int r, int c)
{
	if (r<0 || r>n + 1 || c<0 || c>n + 1 || a[r][c] != 0) return;
	a[r][c] = -1;
	for (int i = 1; i <= 4; i++) dfs(r + x[i], c + y[i]);
}

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);

	std::cin >> n;
	for (int i = 1; i <= n; i++)
		for (int j = 1; j <= n; j++)
			std::cin >> a[i][j];

	dfs(0, 0);

	for (int i = 1; i <= n; i++) {
		for (int j = 1; j <= n; j++) {
			if (a[i][j] == -1) std::cout << 0 << ' ';
			else if (a[i][j] == 1) std::cout << 1 << ' ';
			else if (!a[i][j]) std::cout << 2 << ' ';
		}
		std::cout << '\n';
	}

	return 0;
}