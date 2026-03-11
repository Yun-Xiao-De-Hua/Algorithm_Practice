//#include<iostream>
//#include<climits>
//
//const int N = 15;
//int n, S[N], B[N], ans = INT_MAX;
//
//void dfs(int i, int s, int b) {
//	if (i > n) {
//		if (s == 1 && b == 0) return;
//		ans = std::min(ans, std::abs(s - b));
//		return;
//	}
//
//	dfs(i + 1, s * S[i], b + B[i]);
//	dfs(i + 1, s, b);
//}
//
//int main()
//{
//	std::ios::sync_with_stdio(false);
//	std::cin.tie(0);
//
//	std::cin >> n;
//	for (int i = 1; i <= n; i++) std::cin >> S[i] >> B[i];
//
//	dfs(1, 1, 0);
//
//	std::cout << ans;
//
//	return 0;
//}

#include<iostream>
#include<climits>

const int N = 15;
int n, S[N], B[N], ans = INT_MAX;

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);

	std::cin >> n;
	for (int i = 0; i < n; ++i) std::cin >> S[i] >> B[i];

	for (int i = 1; i < (1 << n); i++) {
		int s = 1, b = 0;
		for (int j = 0; j < n; j++) {
			if ((i >> j)&1) {
				s *= S[j];
				b += B[j];
			}
		}
		ans = std::min(ans, std::abs(s - b));
	}

	std::cout << ans;

	return 0;
}