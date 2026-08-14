#include<iostream>
#include<vector>
using namespace std;

class Solution {
public:
    void dfs(vector<vector<int>>& res, vector<int>& output, const vector<int>& nums, int pos, int end)
    {
        if (pos == end) {
            res.emplace_back(output);
            return;
        }

        output.push_back(nums[pos]);
        dfs(res, output, nums, pos + 1, end);
        output.pop_back();

        dfs(res, output, nums, pos + 1, end);
    }

    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>>res;
        vector<int> output;
        dfs(res, output, nums, 0, nums.size());

        return res;
    }
};

int main()
{
	return 0;
}