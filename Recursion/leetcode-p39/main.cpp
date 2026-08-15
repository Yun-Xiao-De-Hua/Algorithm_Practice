#include<iostream>
#include<vector>
using namespace std;

class Solution {
public:
    vector<vector<int>> res;
    int n, t;

    void dfs(const vector<int>& can, vector<int>& tem, int sum, int idx)
    {

        if (sum >= t) {
            if (sum == t) res.emplace_back(tem);
            return;
        }

        for (int i = idx; i < n; ++i) {
            tem.push_back(can[i]);
            dfs(can, tem, sum + can[i], i);
            tem.pop_back();
        }
    }

    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        n = candidates.size();
        t = target;
        vector<int> mid;
        dfs(candidates, mid, 0, 0);

        return res;
    }
};

int main()
{
	return 0;
}