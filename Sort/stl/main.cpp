#include<iostream>
#include<algorithm>
#include<vector>

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	int n; std::cin >> n;
	
	std::vector<int> a;
	for (int i = 1; i <= n; i++) {
		int num; std::cin >> num;
		a.push_back(num);
	}

	std::sort(a.begin(), a.end());
	a.erase(std::unique(a.begin(), a.end()), a.end());

	for (auto& v : a) std::cout << v << " ";

	return 0;
}