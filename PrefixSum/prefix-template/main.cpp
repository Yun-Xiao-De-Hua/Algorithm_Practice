#include<iostream>

const int N = 1e5 + 10;
using ll = long long;
ll a[N], prefix[N];

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	int t; std::cin >> t;

	while (t--) {
		int n, q; std::cin >> n >> q;
		for (int i = 1; i <= n; i++) std::cin >> a[i];
		for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + a[i];

		while (q--) {
			int l, r; std::cin >> l >> r;
			std::cout << prefix[r] - prefix[l - 1] << '\n';
		}
	}

	return 0;
}