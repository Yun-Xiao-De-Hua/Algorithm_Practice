#include<iostream>
#include<algorithm>

int n, dp[32], ans;

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);

	std::cin >> n;
	while (n--) {
		int a;
		std::cin >> a;

		int maxn = 1;
		for (int i = 0; i <= 30; i++) {
			if ((1 << i) & a) maxn = std::max(maxn, dp[i] + 1);
		}

		for (int i = 0; i <= 30; i++) {
			if ((1 << i) & a) dp[i] = std::max(maxn, dp[i]);
		}

		ans = std::max(ans, maxn);
	}

	std::cout << ans;

	return 0;
}