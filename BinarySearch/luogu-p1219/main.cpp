#include<iostream>

int n, col[20], dial[50], diar[50], ans[20], solution_cnt;

void dfs(int row) {
	if (row > n) {
		if (++solution_cnt <= 3) {
			for (int i = 1; i <= n; i++) {
				std::cout << ans[i] << (i == n ? " \n" : " ");
			}
		}
		return;
	}

	for (int i = 1; i <= n; i++) {
		if (!col[i] && !dial[row - i + n] && !diar[row + i]) {
			ans[row] = i;
			col[i] = dial[row - i + n] = diar[row + i] = 1;
			dfs(row + 1);
			col[i] = dial[row - i + n] = diar[row + i] = 0;
		}
	}
}

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);

	std::cin >> n;

	dfs(1);

	std::cout << solution_cnt;

	return 0;
}