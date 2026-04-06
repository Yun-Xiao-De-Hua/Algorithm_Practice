// 贪心思想，逻辑上优先处理列，再处理行；实现上放置在一次遍历中进行状态转移和结果更新
// state：（'#','.'）:1	（'.','#'）:2	（'#','#'）：3

#include<iostream>
#include<string>

const int N = 1e6 + 5;
std::string a, b;
int ans;

void solve()
{
	int last_pos = -1, last_state = -1;
	for (int i = 0; i < a.length(); i++) {
		if (a[i] == '.' && b[i] == '.') continue;
		if (last_pos != -1) ans += i - last_pos - 1;

		if (a[i] == '#' && b[i] == '#') last_state = 3;
		else if (a[i] == '#' && b[i] == '.') {
			if (last_state == 2) {
				ans++;
				last_state = 3;
			}
			else last_state = 1;
		}
		else if (a[i] == '.' && b[i] == '#') {
			if (last_state == 1) {
				ans++;
				last_state = 3;
			}
			else last_state = 2;
		}

		last_pos = i;
	}
}

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	std::cin >> a >> b;

	solve();

	std::cout << ans;

	return 0;
}