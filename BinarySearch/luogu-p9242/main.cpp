#include<iostream>
#include<string>
#include<algorithm>

int n, maxn, dp[10];

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);

	std::cin >> n;

	std::string s;
	for (int i = 1; i <= n; i++) {
		std::cin >> s;
		int l = s.length();
		dp[s[l - 1] - '0'] = std::max(dp[s[0] - '0'] + 1, dp[s[l - 1] - '0']);
	}

	for (int i = 0; i < 10; i++) maxn = std::max(maxn, dp[i]);

	std::cout << n - maxn;

	return 0;
}