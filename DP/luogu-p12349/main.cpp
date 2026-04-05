// 线性DP
// 注意处理输入的时候，输入的整数有空格和没有空格的区别，有空格是分别输入，没空格是作为一个整体！

#include<iostream>
#include<algorithm>
#include<string>

int n, m, a[1005][1005], dp[1005][2];
std::string input;

int get_value(int r, int flip1, int flip2)
{
	int sum = 0;
	for (int c = 1; c <= m; c++) {
		int cnt = 0;
		if (r > 1) cnt += ((a[r][c] ^ flip1) == a[r - 1][c]);
		if (r < n) cnt += ((a[r][c] ^ flip2) == a[r + 1][c]);
		if (c > 1) cnt += (a[r][c] == a[r][c - 1]);
		if (c < m) cnt += (a[r][c] == a[r][c + 1]);
		sum += cnt * cnt;
	}

	return sum;
}

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	std::cin >> n >> m;
	for (int i = 1; i <= n; i++) {
		std::cin >> input;
		for (int j = 1; j <= m; j++) {
			a[i][j] = input[j-1] - '0';
		}
	}

	dp[2][0] = get_value(1, 0, 0);
	dp[2][1] = get_value(1, 0, 1);
	for (int i = 3; i <= n + 1; i++) {
		dp[i][0] = std::max(dp[i - 1][0] + get_value(i - 1, 0, 0), dp[i - 1][1] + get_value(i - 1, 1, 0));
		dp[i][1] = std::max(dp[i - 1][0] + get_value(i - 1, 0, 1), dp[i - 1][1] + get_value(i - 1, 1, 1));
	}

	std::cout << std::max(dp[n + 1][1], dp[n + 1][0]);

	return 0;
}