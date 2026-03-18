#include<iostream>
#include<utility>
#include<climits>
#include<queue>
#include<set>

int n, a, b, k[210], ans = INT_MAX;
std::queue<std::pair<int, int>> q;
std::set<int> visited;

void bfs(int start)
{
	std::pair<int, int> p{ start,0 };
	q.push(p);

	while (!q.empty()) {
		std::pair<int, int> s = q.front();
		q.pop();

		int pos = s.first;
		int step = s.second;

		if (visited.find(pos) != visited.end()) continue;
		visited.insert(pos);

		if (pos == b) {
			ans = step;
			return;
		}

		step++;

		int up = pos + k[pos];
		int down = pos - k[pos];

		if (down >= 1) {
			std::pair<int, int> s1{down,step};
			q.push(s1);
		}

		if (up <= n) {
			std::pair<int, int> s2{up,step};
			q.push(s2);
		}
	}
}

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);

	std::cin >> n >> a >> b;
	for (int i = 1; i <= n; i++) std::cin >> k[i];

	bfs(a);

	if (ans != INT_MAX) std::cout << ans;
	else std::cout << -1;

	return 0;
}