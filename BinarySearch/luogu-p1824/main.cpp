#include<iostream>
#include<algorithm>

const int N = 2e5 + 10;
int a[N];
int n, m;

int get_max_num(int dist)
{
	int res = 0, pre_idx = -1e9;
	for (int i = 1; i <= n; i++) {
		if (a[i] - pre_idx >= dist) {
			res++;
			pre_idx = a[i];
		}
	}

	return res;
}

int solve(int num)
{
	int l = 1, r = 1e9 + 10;
	while (l + 1 != r) {
		int mid = l + (r - l) / 2;
		if (get_max_num(mid) >= num) l = mid;
		else r = mid;
	}

	return l;
}

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	std::cin >> n >> m;

	for (int i = 1; i <= n; i++) std::cin >> a[i];
	std::sort(a + 1, a + 1 + n);

	std::cout << solve(m);

	return 0;
}