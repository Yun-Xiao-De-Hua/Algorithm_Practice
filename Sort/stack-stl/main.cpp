#include<iostream>
#include<stack>

const int N = 1e5 + 5;
int a[N], n;

void solve()
{
	std::stack<int> stk;
	int pos = 1;
	for (int i = 1; i <= n; ++i) {
		while (pos < n + 1 && (stk.empty() || stk.top() != i)) stk.push(a[pos++]);

		if (stk.top() != i) {
			std::cout << "No" << '\n';
			return;
		}
		else stk.pop();
	}

	std::cout << "Yes" << '\n';
}

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	std::cin >> n;
	for (int i = 1; i <= n; ++i) std::cin >> a[i];

	solve();

	return 0;
}