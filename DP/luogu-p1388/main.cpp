// 区间DP
// 维护一个数组，记录区间[i,j]使用p个乘号的最大值
// 初始化时，对于使用0个乘号的区间(即区间和),使用前缀和优化计算

#include<iostream>
#include<algorithm>

int n, k, f[20][20][20], s[20];

int main() 
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);

	std::cin >> n >> k;

	for (int i = 1; i <= n; i++) {
		std::cin >> s[i];
		s[i] += s[i - 1];
	}

	for (int i = 1; i <= n; i++)
		for (int j = i; j <= n; j++) f[i][j][0] = s[j] - s[i - 1];

	for (int r = 2; r <= n; r++) {
		for (int i = 1; i + r - 1 <= n; i++) {
			int j = i + r - 1;

			for (int p = 1; p <= k; p++) {
				for (int m = i; m <= j - 1; m++) { // 通过枚举分割点，实现不同括号添加方式的遍历

					for (int q = 0; q <= p && q <= m - i; q++) {

						if (p - q >= 0 && p - q <= j - m - 1) // 变量范围检查
							f[i][j][p] = std::max(f[i][j][p], f[i][m][q] + f[m + 1][j][p - q]);

						if (p - q - 1 >= 0 && p - q - 1 <= j - m - 1)
							f[i][j][p] = std::max(f[i][j][p], f[i][m][q] * f[m + 1][j][p - q - 1]);
					}
				}
			}
		}
	}

	std::cout << f[1][n][k];

	return 0;
}