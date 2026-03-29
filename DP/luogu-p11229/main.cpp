// DP思路
// 维护一个数组，表示使用i根棒能够拼出的最小数
// 根据转态转移方程，每个状态之和临近几个状态有关，考虑使用滚动数组优化空间复杂度
// 可能会超过long long表示范围，考虑使用string模拟大整数

#include<iostream>
#include<string>
#include<algorithm>

int n, t, maxn;
// 数字：0,1,2,3,4,5,6,7,8,9
// 棒数：6,2,5,5,4,5,6,3,7,6
std::string mn[8] = { "-1","-1","1","7","4","2","6","8" };
std::string dp[10], ans[55];
int num[55];

std::string comp(const std::string& a, const std::string& b)
{
	if (a.length() != b.length()) return a.length() < b.length() ? a : b;
	else return a < b ? a : b;
}

void solve(int n)
{
	dp[0] = dp[1] = "-1";

	for (int i = 2; i <= n; i++) {
		dp[i % 8] = dp[(i - 2 + 8) % 8] + mn[2];
		if (i <= 7) dp[i % 8] = mn[i];

		for (int j = i - 2; j >= i - 7 && j >= 0; j--) {
			if (dp[j % 8] != "-1") dp[i % 8] = comp(dp[i % 8], dp[j % 8] + mn[i - j]);
			if (dp[j % 8] != "-1" && dp[j % 8] != "0" && i - j == 6) dp[i % 8] = comp(dp[i % 8], dp[j % 8] + "0");
		}

		for (int j = 1; j <= t; j++)
			if (num[j] == i) ans[j] = dp[i % 8];
	}
}

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	std::cin >> t;
	for (int i = 1; i <= t; i++) {
		std::cin >> num[i];
		maxn = std::max(maxn, num[i]);
	}

	solve(maxn);

	for (int i = 1; i <= t; i++) {
		if (ans[i] == "") std::cout << -1 << '\n';
		else std::cout << ans[i] << '\n';
	}

	return 0;
}