#include<iostream>
#include<vector>
#include<string>
using namespace std;

class Solution {
public:
    vector<string> res;

    void dfs(string& tem, int l, int r, int dep, int n)
    {
        if (dep == 2 * n) {
            res.emplace_back(tem);
            return;
        }

        if (l < n) {
            tem.push_back('(');
            dfs(tem, l + 1, r, dep + 1, n);
            tem.pop_back();
        }
        if (r < l) {
            tem.push_back(')');
            dfs(tem, l, r + 1, dep + 1, n);
            tem.pop_back();
        }
    }

    vector<string> generateParenthesis(int n) {
        string s;
        dfs(s, 0, 0, 0, n);
        return res;
    }
};


int main()
{
	return 0;
}