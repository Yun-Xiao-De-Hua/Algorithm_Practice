#include<iostream>
using namespace std;
using ll = long long;

const int N = 1e6 + 10;
ll tree[N], n, m, total = 0;

bool check(ll h)
{
	ll sup = 0;
	for (int i = 1; i <= n; ++i) 
		if (h > tree[i]) sup += (h - tree[i]);

	return (total - h * n + sup >= m) ? 1 : 0;
}

ll search(int l, int r)
{
	while (l + 1 != r) {
		ll mid = l + (r - l) / 2;
		if (check(mid)) l = mid;
		else r = mid;
	}

	return l;
}

int main()
{
	ios::sync_with_stdio(0);
	cin.tie(0);

	cin >> n >> m;

	ll r = 0;
	for (int i = 1; i <= n; ++i) {
		cin >> tree[i];
		if (tree[i] > r) r = tree[i];
		total += tree[i];
	}

	cout << search(0, r + 1);

	return 0;
}