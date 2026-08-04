#include<iostream>
#include<vector>
using namespace std;

vector<int> path;
int n;

void dfs(int rem, int min)
{
	if (rem == 0) {
		if (path.size() > 1) {
			for (int i = 0; i < path.size(); ++i) cout << path[i] << ((i != path.size() - 1) ? "+" : "");
			cout << '\n';
		}
	}

	for (int i = min; i <= rem; ++i) {
		if (i >= n) continue;

		path.push_back(i);
		dfs(rem - i, i);
		path.pop_back();
	}
}

int main()
{
	ios::sync_with_stdio(0);
	cin.tie(0);

	cin >> n;

	dfs(n, 1);

	return 0;
}