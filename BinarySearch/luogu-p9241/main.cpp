#include<iostream>
#include<algorithm>

const int N = 15;

struct plane {
	int e, l, t;
}p[N];

int T, n, isl[N];
bool flag;

void dfs(int dep, int now) {
	if (flag) return;
	if (dep > n) {
		flag = true;
		return;
	}

	for (int i = 1; i <= n; i++) {
		if (!isl[i] && p[i].l < now) return;
		if (!isl[i] && p[i].l >= now) {
			isl[i] = 1;
			dfs(dep + 1, std::max(p[i].e, now) + p[i].t);
			isl[i] = 0;
		}
		if (flag) return;
	}
}

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);

	std::cin >> T;
	while (T--) {
		std::cin >> n;
		for (int i = 1; i <= n; i++) {
			int d;
			std::cin >> p[i].e >> d >> p[i].t;
			p[i].l = p[i].e + d;
		}

		flag = false;
		dfs(1, 0);

		if (flag) std::cout << "YES\n";
		else std::cout << "NO\n";
	}

	return 0;
}