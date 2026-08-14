#include<iostream>
#include<vector>
#include<utility>
using namespace std;

class Solution {
public:

    void search(vector<vector<int>>& res, vector<int>& output,int pos,int end) {
        if (pos == end) {
            res.emplace_back(output);
            return;
        }

        for (int i = pos; i < end; ++i) {
            swap(output[pos], output[i]);
            search(res, output, pos + 1, end);
            swap(output[pos], output[i]);
        }
    }

    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> res;
        search(res, nums, 0, nums.size());
        return res;
    }
};

int main()
{
	return 0;
}