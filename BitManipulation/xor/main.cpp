#include<iostream>

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	int t; std::cin >> t;
	
	while (t--) {
		int n; std::cin >> n;

		int ans = 0;
		for (int i = 1; i <= n; i++) {
			int num; std::cin >> num;
			ans ^= num;
		}

		std::cout << ans << "\n";
	}

	return 0;
}