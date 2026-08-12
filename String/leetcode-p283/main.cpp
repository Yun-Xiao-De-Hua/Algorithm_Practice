#include<vector>
#include<utility>
using namespace std;


class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        if (nums.size() == 1) return;

        int n = nums.size(), l = 0, r = 0;
        while (r < n) {
            if (nums[r]) swap(nums[l++], nums[r]);
            r++;
        }
    }
};

int main()
{
    return 0;
}