// 考虑使用邻接矩阵来表示连通关系
// 路径记录与更新，可以通过开两个空间，一个记录过程变化，一个结合判定条件进行更新
// 遍历所有可能动作进行搜索时，注意状态回溯

#include<iostream>
#include<vector>

int d[25], linked[25][25], n, maxn;
std::vector<int> cur;
std::vector<int> ans;

void dfs(int now, int sum)
{
	cur.push_back(now);

	bool has_next = false;
	for (int i = now + 1; i <= n; i++) {
		if (linked[now][i]) {
			dfs(i, sum + d[i]);
			has_next = true;
		}
	}

	if (!has_next) {
		if (sum > maxn) {
			maxn = sum;
			ans = cur;
		}
	}

	cur.pop_back();
}

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	std::cin >> n;
	for (int i = 1; i <= n; i++) std::cin >> d[i];

	for (int i = 1; i < n; i++)
		for (int j = i + 1; j <= n; j++)
			std::cin >> linked[i][j];

	for (int i = 1; i <= n; i++) dfs(i, d[i]);

	for (int i = 0; i < ans.size(); i++)
		std::cout << ans[i] << (i != ans.size() - 1 ? " " : "");

	std::cout << "\n" << maxn;

	return 0;
}