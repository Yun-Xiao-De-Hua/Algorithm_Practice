#include<iostream>
#include<string>

using namespace std;

string s;
long long ans;
int brkpts, l, r;

int main()
{
	ios::sync_with_stdio(0);
	cin.tie(0);

	cin >> s;

	for (int i = 0; i < s.size(); i++) {
		while (brkpts > 1) {
			l++;
			if (s[l - 1] != s[l] && s[l] - s[l - 1] != 1) brkpts--;
		}

		ans += r - l + 1;

		r++;
		if(s[r - 1] != s[r] && s[r] - s[r - 1] != 1) brkpts++;
	}

	cout << ans;

	return 0;
}