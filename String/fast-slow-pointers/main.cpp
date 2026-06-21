#include<iostream>

const int N = 1e5 + 5;
int a[N], b[N];

void solve() {
	int n; std::cin >> n;
	for (int i = 1; i <= n; i++) std::cin >> a[i];
	for (int i = 1; i <= n; i++) b[a[i]] = 0;

	int ans = 0;
	for (int i = 1, j = 0; i <= n; i++) {
		while (j < n && !b[a[j + 1]]) b[a[++j]]++;
		ans = std::max(ans, j - i + 1);
		b[a[i]]--;
	}

	std::cout << ans << '\n';
}

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	int t; std::cin >> t;

	while (t--) solve();

	return 0;
}