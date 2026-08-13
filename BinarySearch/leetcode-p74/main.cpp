#include<iostream>
#include<vector>
using namespace std;

class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size(), n = matrix[0].size();
        int l = 0, r = m * n;

        while (l + 1 != r) {
            int mid = l + (r - l) / 2;
            int mid_r = mid / n;
            int mid_c = mid % n;

            if (matrix[mid_r][mid_c] <= target) l = mid;
            else r = mid;
        }

        return matrix[l / n][l % n] == target;
    }
};


int main()
{
	return 0;
}