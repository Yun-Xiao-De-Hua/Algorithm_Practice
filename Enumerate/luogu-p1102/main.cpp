//#include<iostream>
//#include<set>
//
//const int N = (int)1e6;
//int n, c, ans, a[N];
//std::multiset<int> s;
//
//int main()
//{
//	std::ios::sync_with_stdio(false);
//	std::cin.tie(0);
//
//	std::cin >> n >> c;
//
//	for (int i = 1; i <= n; i++) {
//		std::cin >> a[i];
//		s.insert(a[i]);
//	}
//
//	for (int i = 1; i <= n; i++) {
//		ans += s.count(a[i] + c);
//	}
//
//	std::cout << ans;
//
//	return 0;
//}

#include<iostream>
#include<algorithm>

typedef long long ll;

const int N = 2e5 + 5;
int n, c, a[N];
ll ans;

int main()
{
	std::cin >> n >> c;
	for (int i = 1; i <= n; i++) std::cin >> a[i];

	std::sort(a + 1, a + 1 + n);

	auto it = std::upper_bound(a + 1, a + 1 + n, c);
	for (int i = 1; i <= n; i++) 
		ans += (std::upper_bound(a + 1, a + 1 + n, a[i] + c) - std::lower_bound(a + 1, a + 1 + n, a[i] + c));

	std::cout << ans;

	return 0;
}