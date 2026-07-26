#include<iostream>

const int N = 2e5 + 10;
int a[N], n;

void solve(int x)
{
	int l = 0, r = n;
	while (l + 1 != r) {
		int mid = l + (r - l) / 2;
		if (a[mid] < x) l = mid;
		else r = mid;
	}

	if (a[r] == x) std::cout << r << ' ';
	else std::cout << -1 << ' ';
}

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	int q;
	std::cin >> n >> q;

	for (int i = 1; i <= n; i++) std::cin >> a[i];

	while (q--)
	{
		int x; std::cin >> x;
		solve(x);
	}

	return 0;
}