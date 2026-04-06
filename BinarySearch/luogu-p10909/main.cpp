// 求解结果具有单调性，考虑使用二分
// 学习到将题设中的特殊性质转换为等价的数学表达：有一次机会跳两倍距离，等价于多增加一个检查点

#include<iostream>

const int N = 1e5 + 5;
int n, m, a[N];

bool meet(int l)
{
	int points = 0;
	for (int i = 1; i <= n; i++) {
		int dis = a[i] - a[i - 1];
		points += (dis + l - 1) / l - 1;
	}
	return points <= m + 1;
}

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	std::cin >> n >> m;
	for (int i = 1; i <= n; i++) std::cin >> a[i];

	int l = 1, r = a[n];
	while (l < r) {
		int mid = l + (r - l) / 2;
		if (meet(mid)) r = mid;
		else l = mid + 1;
	}
	
	std::cout << l;

	return 0;
}