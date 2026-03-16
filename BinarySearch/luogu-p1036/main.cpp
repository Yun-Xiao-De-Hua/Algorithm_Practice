#include<iostream>

const int N = 25;
int n, k, a[N], ans;

bool isPrime(int x) {
	if (x <= 1) return false;
	for (int i = 2; i * i <= x; ++i) {
		if (x % i == 0) return false;
	}
	return true;
}

void dfs(int cnt, int sum, int index) {
	if (cnt == k) {
		if (isPrime(sum)) ans++;
		return;
	}

	for (int i = index; i <= n - k + cnt + 1; ++i)
		dfs(cnt + 1, sum + a[i], i + 1);
}

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);

	std::cin >> n >> k;
	for (int i = 1; i <= n; i++) std::cin >> a[i];

	dfs(0, 0, 1);

	std::cout << ans;

	return 0;
}