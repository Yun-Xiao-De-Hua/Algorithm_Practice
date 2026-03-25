// 对区间和进行操作，考虑使用前缀和进行优化
// 遍历两个端点时间复杂度为O(n^2)，无法通过
// 考虑使用模的性质：两个数对m同余，那么m能够整除两个数的差
// 因此维护两个数组，分别记录同一余数最先和最晚出现的位置，差值即为候选最大值


//#include<iostream>
//#include<algorithm>
//
//typedef long long ll;
//const int N = 5e4 + 10;
//int n, ans;
//ll s[N];
//
//int main()
//{
//	std::ios::sync_with_stdio(false);
//	std::cin.tie(0);
//
//	std::cin >> n;
//	for (int i = 1; i <= n; i++) {
//		std::cin >> s[i];
//		s[i] += s[i - 1];
//	}
//
//	for (int i = 1; i <= n; i++)
//		for (int j = i; j <= n; j++) {
//			if ((s[j] - s[i - 1]) % 7 == 0) ans = std::max(ans, j - i + 1);
//		}
//
//	std::cout << ans;
//
//	return 0;
//}

#include<iostream>
#include<algorithm>

const int N = 5e4 + 10;
int n, ans, s[N], f[7], l[7];

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);

	std::cin >> n;
	for (int i = 1; i <= n; i++) {
		std::cin >> s[i];
		s[i] = (s[i] + s[i - 1]) % 7;
	}

	for (int i = n; i >= 1; i--) f[s[i]] = i;
	f[0] = 0;
	for (int i = 1; i <= n; i++) l[s[i]] = i;

	for (int i = 0; i <= 6; i++) ans = std::max(l[i] - f[i], ans);

	std::cout << ans;

	return 0;
}