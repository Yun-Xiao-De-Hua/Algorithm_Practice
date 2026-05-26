#include<iostream>

const int N = 1e5 + 10;
using ll = long long;
ll a[N], diff[N], prefix[N];

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	int n, p, q; std::cin >> n >> p >> q;

	for (int i = 1; i <= n; i++) std::cin >> a[i];
	for (int i = 1; i <= n; i++) diff[i] = a[i] - a[i - 1];
	
	while (p--) {
		int l, r, x; std::cin >> l >> r >> x;
		diff[l] += x;
		diff[r + 1] -= x;
	}

	for (int i = 1; i <= n; i++) a[i] = diff[i] + a[i - 1];
	for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + a[i];

	while (q--) {
		int l, r; std::cin >> l >> r;
		std::cout << prefix[r] - prefix[l - 1] << '\n';
	}

	return 0;
}