#include<iostream>
#include<vector>
#include<cstring>
using namespace std;

class Solution {
public:
    int dx[5] = { 0,0,1,0,-1 };
    int dy[5] = { 0,1,0,-1,0 };
    static const int N = 6;
    int visited[N][N] = {};

    bool dfs(const vector<vector<char>>& board, const string& word, int x, int y, int dep)
    {
        if (board[x][y] != word[dep]) return false;
        else if (dep == word.size() - 1) return true;

        visited[x][y] = 1;
        bool flag = false;
        for (int i = 1; i <= 4; ++i) {
            int newx = x + dx[i];
            int newy = y + dy[i];

            if (newx >= 0 && newx < board.size() && newy >= 0 && newy < board[0].size()) {
                if (!visited[newx][newy]) {
                    flag = dfs(board, word, newx, newy, dep + 1);
                    if (flag) break;
                }
            }
        }
        visited[x][y] = 0;

        return flag;
    }

    bool exist(vector<vector<char>>& board, string word) {
        int m = board.size();
        int n = board[0].size();

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                bool check = dfs(board, word, i, j, 0);
                if (check) return true;
            }
        }

        return false;
    }
};

int main()
{
	return 0;
}