#include<iostream>
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
}