#include<iostream>
using namespace std;

const int N = 1e6 + 10;
int a[N];

int search(int l, int r, int t)
{
	while (l + 1 != r) {
		int mid = l + (r - l) / 2;
		if (a[mid] < t) l = mid;
		else r = mid;
	}

	if (a[r] == t) return r;
	else return -1;
}

int main()
{
	ios::sync_with_stdio(0);
	cin.tie(0);

	int n, m; cin >> n >> m;
	for (int i = 1; i <= n; ++i) cin >> a[i];

	while (m--) {
		int t; cin >> t;
		cout << search(0, n, t) << ' ';
	}

	return 0;
}