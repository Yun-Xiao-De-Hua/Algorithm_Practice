#include<iostream>
#include<vector>
using namespace std;

class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        if (target < nums[0]) return 0;
        if (target > nums[nums.size() - 1]) return nums.size();

        int l = 0, r = nums.size();

        while (l + 1 != r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] <= target) l = mid;
            else r = mid;
        }

        return nums[l] == target ? l : r;
    }
};



int main()
{
	return 0;
}