#include<iostream>
#include<vector>
#include<string>
using namespace std;

class Solution {
public:
    string a[10] = { "","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz" };
    vector<string> res;
    int n;
    string s;

    void dfs(int pos, string t)
    {
        if (pos == n) {
            res.push_back(t);
            return;
        }

        for (char& c : a[s[pos] - '0']) {
            dfs(pos + 1, t + c);
        }
    }

    vector<string> letterCombinations(string digits) {
        n = digits.size();
        s = digits;
        dfs(0, "");
        return res;
    }
};

int main()
{
	return 0;
}