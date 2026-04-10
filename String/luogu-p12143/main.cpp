// 学习到一个思路：除了纵向思考子串数量，也可以横向思考，即考虑以每个索引位置结束的子串一共有多少个
// 因为每一个子串都是由一个左索引和一个右索引组成的，遍历所有的索引位置，一定统计的所有的情况
// 通过滑动窗口和双指针实现，时间复杂度为O(n)

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