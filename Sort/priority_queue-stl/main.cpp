#include<iostream>
#include<queue>
#include<vector>
using ll = long long;

struct cmp {
	bool operator()(const ll& n1, const ll& n2)const
	{
		return n1 < n2;
	}
};

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	int q; std::cin >> q;

	std::priority_queue<ll, std::vector<ll>, cmp> pq;
	ll sum = 0;
	while (q--) {
		int op; std::cin >> op;
		if (op == 1) {
			int x; std::cin >> x;
			sum += x;
			pq.push(x);
		}
		else if (op == 2) {
			if (!pq.empty()) {
				sum -= pq.top();
				pq.pop();
			}
		}
	}

	std::cout << sum << '\n';

	return 0;
}