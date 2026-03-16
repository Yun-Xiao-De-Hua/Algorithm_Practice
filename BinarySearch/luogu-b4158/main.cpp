#include<iostream>
#include<string>
#include<algorithm>

bool found = false;
std::string s;
int t;

bool isPrime(int x) {
	if (x <= 1) return false;
	for (int i = 2; i * i <= x; i++)
		if (x % i == 0) return false;
	return true;
}

void dfs(int dep) {
	if (dep == s.size()) {
		int num = std::stoi(s);
		if (isPrime(num)) {
			std::cout << num << '\n';
			found = true;
		}
		return;
	}

	if (s[dep] != '*') dfs(dep + 1);
	else {
		for (int i = 0; i <= 9; i++) {
			s[dep] = i + '0';
			dfs(dep + 1);
			if (found) return;	// ¼ôÖ¦
		}
		s[dep] = '*';
	}
}

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);

	std::cin >> t;
	while (t--) {
		std::cin >> s;
		dfs(0);
		if (!found) std::cout << -1 << '\n';
		found = false;
	}

	return 0;
}