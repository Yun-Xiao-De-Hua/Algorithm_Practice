// 搜索问题，排序+二分；
// 或者使用stl容器建立映射关系，使用哈希表可实现在O(1)时间内完成搜索

#include<iostream>
#include<map>

int n, Q, m, a;
std::map<int, int> t;

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);

	std::cin >> n;
	for (int i = 1; i <= n; i++) {
		std::cin >> a;
		t.insert({ a,i });
	}

	std::cin >> Q;
	while (Q--) {
		std::cin >> m;
		if (t.find(m) != t.end()) std::cout << t[m] << '\n';
		else std::cout << 0 << '\n';
	}

	return 0;
}