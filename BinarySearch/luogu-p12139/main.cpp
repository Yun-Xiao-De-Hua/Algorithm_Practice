// 学习到一个思路：二维矩阵可以在逻辑上转换为线性数组的形式进行遍历；
// 通过维护一个线性数组索引，可得出该索引对应的横纵坐标；

#include<iostream>
#include<string>
#include<unordered_set>
using namespace std;

int board[6][6] = { 
	{1,0,-1,0,-1,-1}, 
	{-1,-1,-1,0,-1,-1},
	{-1,-1,-1,-1,0,0},
	{-1,-1,-1,-1,-1,-1},
	{-1,-1,1,-1,-1,1},
	{-1,0,-1,-1,1,-1} 
};

int row_cnt[6][2], col_cnt[6][2];

bool is_unique() {
	for (int i = 0; i < 6; i++) {
		for (int j = i + 1; j < 6; j++) {
			bool same = true;
			for (int k = 0; k < 6; k++) {
				if (board[i][k] != board[j][k]) {
					same = false;
					break;
				}
			}
			if (same) return false;
		}
	}

	for (int i = 0; i < 6; i++) {
		for (int j = i + 1; j < 6; j++) {
			bool same = true;
			for (int k = 0; k < 6; k++) {
				if (board[k][i] != board[k][j]) {
					same = false;
					break;
				}
			}
			if (same) return false;
		}
	}
	return true;
}

void dfs(int idx){
	if (idx == 36) {
		if (is_unique()) {
			string res = "";
			for (int i = 0; i < 6; i++)
				for (int j = 0; j < 6; j++)
					res += to_string(board[i][j]);
			cout << res;
			exit(0);
		}
		return;
	}

	int r = idx / 6;
	int c = idx % 6;

	if (board[r][c] != -1) {
		dfs(idx + 1);
		return;
	}

	for (int i = 0; i <= 1; i++) {
		if (row_cnt[r][i] >= 3 || col_cnt[c][i] >= 3) continue;

		if (r >= 2 && board[r - 1][c] == i && board[r - 2][c] == i) continue;
		if (c >= 2 && board[r][c - 1] == i && board[r][c - 2] == i) continue;
		
		row_cnt[r][i]++;
		col_cnt[c][i]++;
		board[r][c] = i;

		dfs(idx + 1);

		row_cnt[r][i]--;
		col_cnt[c][i]--;
		board[r][c] = -1;
	}
}


int main()
{
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 6; j++) {
			if (board[i][j] != -1) {
				row_cnt[i][board[i][j]]++;
				col_cnt[j][board[i][j]]++;
			}
		}
	}

	dfs(0);

	return 0;
}