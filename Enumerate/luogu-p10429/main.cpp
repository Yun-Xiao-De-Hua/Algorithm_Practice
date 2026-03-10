/*#include<iostream>
#include<climits>

const int N = 1e3 + 10;
int n,a[N];

int minres(int ans, long long diff) {
	return ans < diff ? ans : diff;
}

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0), std::cout.tie(0);

	std::cin >> n;
	for (int i = 1; i <= n; ++i) std::cin >> a[i];

	int ans = INT_MAX;
	for (int i = 1; i <= n; i++) {
		for (int j = i+1; j <= n; j++) {
			int l = i - 1, r = j + 1;
			long long suml = a[i], sumr = a[j];
			ans = minres(ans, std::abs(suml - sumr));

			while (l >= 1 && r <= n) {
				if (suml < sumr) suml += a[l--];
				else sumr += a[r++];
				ans = minres(ans, std::abs(suml - sumr));
			}

			while (l >= 1) {
				suml += a[l--];
				ans = minres(ans, std::abs(suml - sumr));
			}

			while (r <= n) {
				sumr += a[r++];
				ans = minres(ans, std::abs(suml - sumr));
			}
		}
	}

	std::cout << ans;

	return 0;
}*/

#include<iostream>
#include<climits>
#include<set>

const int N = 1e3 + 10;
long long a[N];
int n, ans = INT_MAX;
std::multiset<long long> S;

int min(int a, long long b) {
	return a < b ? a : b;
}

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0), std::cout.tie(0);

	std::cin >> n;
	for (int i = 1; i <= n; i++) {
		std::cin >> a[i];
		a[i] = a[i - 1] + a[i];
	}

	for (int i = 1; i <= n; i++) {
		for (int j = i; j <= n; j++) {
			S.insert(a[j] - a[i - 1]);
		}
	}

	for (int i = 1; i <= n; i++) {
		for (int j = i; j <= n; j++) {
			auto p = S.find(a[j] - a[i - 1]);
			S.erase(p);
		}

		for (int j = 1; j <= i; j++) {
			long long suml = a[i] - a[j - 1];
			auto p = S.lower_bound(suml);
			if (p != S.end()) ans = min(ans, abs(*p - suml));
			if (p != S.begin()) ans = min(ans, abs(*(--p) - suml));
		}
	}
	
	std::cout << ans;

	return 0;
}