#include<iostream>

int solve(int num) {
	int ans = 0;

	while (num) {
		if (num & 1) ans++;
		num >>= 1;
	}

	return ans;
}

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	int n; std::cin >> n;
	
	while (n--) {
		int num; std::cin >> num;
		std::cout << solve(num) << " ";
	}

	return 0;
}