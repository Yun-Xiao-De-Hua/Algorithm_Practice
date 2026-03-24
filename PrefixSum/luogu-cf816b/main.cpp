// 考察前缀和与差分
// 利用差分对区间进行统一加减，并结合前缀和还原
// 通过额外维护一个前缀和数组记录满足题意的方案，实现对最终结果的O(1)查询

#include<iostream>

const int N = 2e5 + 10;
typedef long long ll;
int n, k, q, l, r, a, b;
ll d[N], c[N];

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);

	std::cin >> n >> k >> q;

	while (n--) {
		std::cin >> l >> r;
		d[l]++, d[r + 1]--;
	}

	for (int i = 1; i < N; i++) {
		d[i] += d[i - 1];
		int is_ac = d[i] >= k ? 1 : 0;
		c[i] = c[i - 1] + is_ac;
	}

	while (q--) {
		std::cin >> a >> b;
		std::cout << c[b] - c[a - 1] << '\n';
	}

	return 0;
}