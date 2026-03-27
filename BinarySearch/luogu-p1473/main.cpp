// 学习到一个思路：将需要维护的中间状态放置在dfs的形参中进行传递，可自然地实现回溯

#include<iostream>
#include<string>
#include<cctype>

int n;

void dfs(int dep, std::string s)
{
	if (dep >= n) {
		char last = '+';
		int sum = 0, cur = 0;

		for (char c : s) {
			if(c == ' ') continue;
			if (std::isdigit(c)) cur = cur * 10 + c - '0';
			else {
				if (last == '+') sum += cur;
				else sum -= cur;

				last = c;
				cur = 0;
			}
		}

		if (last == '+') sum += cur;
		else sum -= cur;

		if (sum == 0) std::cout << s << '\n';

		return;
	}

	dfs(dep + 1, s + ' ' + std::to_string(dep + 1));

	dfs(dep + 1, s + '+' + std::to_string(dep + 1));

	dfs(dep + 1, s + '-' + std::to_string(dep + 1));
}

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);

	std::cin >> n;
	dfs(1, "1");

	return 0;
}