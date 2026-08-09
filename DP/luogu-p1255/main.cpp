#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

const int N = 5005;
vector<vector<int>> dp(N);

void add(vector<int>& a, vector<int>& b)
{
	int carry = 0;
	for (int i = 0; i < max(a.size(), b.size()) || carry; ++i)
	{
		if (i == a.size()) a.push_back(0);
		int sum = a[i] + (i < b.size() ? b[i] : 0) + carry;
		a[i] = sum % 10;
		carry = sum / 10;
	}
}

int main()
{
	ios::sync_with_stdio(0);
	cin.tie(0);

	int n; cin >> n;

	dp[1] = { 1 };
	dp[2] = { 2 };
	for (int i = 3; i <= n; ++i)
	{
		dp[i] = dp[i - 1];
		add(dp[i], dp[i - 2]);
	}

	for (int i = dp[n].size() - 1; i >= 0; --i) cout << dp[n][i];

	return 0;
}