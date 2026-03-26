// 网格搜索问题
// 处理成染色问题，当出现满足题意的序列，进行染色，后续遍历一遍矩阵，根据染色矩阵数值进行输出
// 利用dfs，当序列满足要求时，再由尾至头进行染色
// 维护一个数组，记录满足题意的起始位置，实现剪枝
// 维护两个数组，依次作为x轴、y轴方向，定义动作

#include<iostream>

char s[105][105], t[8] = { ' ','y','i','z','h','o','n','g' };
int x[9] = { 0, -1, -1, 0, 1, 1, 1, 0, -1};
int y[9] = { 0, 0, -1, -1, -1, 0, 1, 1, 1};
int n, c[105][105], b[105][105];

bool dfs(int x, int y, int h, int r, int next)
{
	if (x <1 || x >n || y <1 || y >n) return 0;

	if (next >= 8) {
		c[x][y] = 1;
		return 1;
	}

	if (s[x+h][y+r]==t[next]) {
		if (dfs(x + h, y + r, h, r, next + 1)) {
			c[x][y] = 1;
			return 1;
		}
	}

	return 0;
}

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);

	std::cin >> n;

	for (int i = 1; i <= n; i++)
		for (int j = 1; j <= n; j++) {
			std::cin >> s[i][j];
			if (s[i][j] == 'y') b[i][j] = 1;
		}

	for (int i = 1; i <= n; i++)
		for (int j = 1; j <= n; j++) {
			if (b[i][j])
				for (int k = 1; k <= 8; k++) dfs(i, j, x[k], y[k], 2);
		}

	for (int i = 1; i <= n; i++) {
		for (int j = 1; j <= n; j++) {
			if (c[i][j]) std::cout << s[i][j];
			else std::cout << '*';
		}
		std::cout << '\n';
	}

	return 0;
}