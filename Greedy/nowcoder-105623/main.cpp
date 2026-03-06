#include<iostream>
#include<string>
#include<vector>

int solve(std::vector<long long>& array, std::string& s, int n) {
	int cnt = 0;

	for (int i = 0; i < n; ++i) {
		if (s[i] == '<' && array[i] >= 0) {
			cnt++;
			array[i] = -1;
		}
		else if (s[i] == '>' && array[i] <= 0) {
			cnt++;
			array[i] = 1;
		}
		else if (s[i] == 'Z' && array[i - 1] * array[i] <= 0) {
			cnt++;
			array[i] = array[i - 1];
		}
	}

	return cnt;
}

int main()
{
	int T, n;
	std::cin >> T;

	for (int i = 0; i < T; ++i) {
		std::cin >> n;

		std::vector<long long>array(n);
		for (int j = 0; j < n; ++j) {
			std::cin >> array[j];
		}

		std::string s;
		std::cin >> s;

		std::cout << solve(array, s, n) << std::endl;
	}

	return 0;
}

