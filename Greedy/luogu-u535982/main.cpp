#include<iostream>
#include<string>
#include<vector>

void solve(int n, std::string& s)
{
	std::string a, b;
	for (int i = 0; i < 2 * n; ++i) {
		if (i & 1) {
			a += 'A';
			b += 'B';
		}
		else {
			a += 'B';
			b += 'A';
		}
	}

	auto calc = [](std::string s, std::string t) -> int {
		int cnt = 0;

		for (int i = 0; i < s.size(); ++i) {
			cnt += (s[i] != t[i]);
		}

		return cnt / 2;
	};

	std::cout << std::min(calc(s, a), calc(s, b)) << std::endl;
}

int main()
{
	int T,n;
	std::string s;
	std::cin >> T;

	for (int i = 0; i < T; ++i) {
		std::cin >> n >> s;
		solve(n, s);
	}

	return 0;
}