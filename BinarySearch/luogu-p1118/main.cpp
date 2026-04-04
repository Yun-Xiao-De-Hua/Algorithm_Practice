// 利用题目数学性质，观察到最后总和中每个参量的权重恰好匹配杨辉三角第N行的数列
// 二维数组打表，存储杨辉三角
// 利用本题求和的单调性进行剪枝
// 学习到一个思路：使用dfs遍历可能的排列时，可以维护一个数组，记录某个数字是否已经被搜索过，从而避免重复

#include<iostream>
#include<cstdlib>

int t[15][15], visited[15], a[15];
int n, m;

void dfs(int dep, int s)
{
	if (s > m) return;
	
	if (dep > n) {
		if (s == m) {
			for (int i = 1; i <= n; i++) std::cout << a[i] << " ";
			exit(0);
		}
		return;
	}

	for (int i = 1; i <= n; i++) {
		if (!visited[i]) {
			visited[i] = 1;
			a[dep] = i;
			dfs(dep + 1, s + t[n][dep] * i);
			visited[i] = 0;
		}
	}
}

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	std::cin >> n >> m;
	
	t[1][1] = 1;
	for (int i = 2; i <= n; i++)
		for (int j = 1; j <= i; j++)
			t[i][j] = t[i - 1][j] + t[i - 1][j - 1];

	dfs(1, 0);

	return 0;
}