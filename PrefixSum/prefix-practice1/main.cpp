#include<iostream>
#include<algorithm>

const int N = 1e5 + 10;
using ll = long long;
ll a[N], w[N], prefix[N];

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	int t; std::cin >> t;
	
	while (t--) {
		int n; std::cin >> n;
		
		for (int i = 1; i <= n; i++) std::cin >> a[i];
		for (int i = 1; i <= n; i++) std::cin >> w[i];

		ll ess = 0;
		for (int i = 1; i <= n; i++) ess += a[i] * w[i];

		for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + (a[i] ? -1 : 1) * w[i];

		ll minl = 0, fix = 0;
		for (int r = 1; r <= n; r++) {
			fix = std::max(fix, prefix[r] - minl);
			minl = std::min(minl, prefix[r]);
		}

		ess += fix;

		std::cout << ess << '\n';
	}

	return 0;
}