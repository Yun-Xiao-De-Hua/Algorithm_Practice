#include<iostream>

typedef long long ll;
const int N = 1e5 + 10;
int s[N], n, k;
ll ans, p[N];

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);

	std::cin >> n >> k;
	for (int i = 1; i <= n; i++) {
		std::cin >> s[i];
		s[i] = (s[i] + s[i - 1]) % k;
	}

	for (int i = 1; i <= n; i++) p[s[i]]++;
	p[0]++;

	for (int i = 0; i <= k - 1; i++)
		if (p[i] > 0) ans += (p[i]) * (p[i] - 1) / 2;

	std::cout << ans;

	return 0;
}