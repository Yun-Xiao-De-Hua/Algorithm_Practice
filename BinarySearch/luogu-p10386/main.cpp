#include<iostream>

const int N = 26;
int board[N], w, b, ans;

bool check(int pos) {
	int r = (pos - 1) / 5 + 1;
	int c = (pos - 1) % 5 + 1;

	int r_sum = 0;
	for (int i = 1; i <= 5; i++) r_sum += board[(r - 1) * 5 + i];
	if (r_sum == 0 || r_sum == 5) return false;

	int c_sum = 0;
	for (int i = 1; i <= 5; i++) c_sum += board[(i - 1) * 5 + c];
	if (c_sum == 0 || c_sum == 5) return false;

	int d1_sum = 0;
	if (r == c) {
		for (int i = 1; i <= 5; i++) d1_sum += board[(i - 1) * 5 + i];
		if (d1_sum == 0 || d1_sum == 5) return false;
	}

	int d2_sum = 0;
	if (r + c == 6) {
		for (int i = 1; i <= 5; i++) d2_sum += board[(i - 1) * 5 + 6 - i];
		if (d2_sum == 0 || d2_sum == 5) return false;
	}

	return true;
}

void dfs(int pos) {
	if (pos == N) {
		ans++;
		return;
	}

	if (w < 13) {
		w++;
		board[pos] = 1;
		if (check(pos)) dfs(pos + 1);
		w--;
	}

	if (b < 12) {
		b++;
		board[pos] = 0;
		if (check(pos)) dfs(pos + 1);
		b--;
	}

	board[pos] = -6;	// »ØËÝ
}

int main()
{
	for (int i = 1; i <= 25; i++) board[i] = -6;

	dfs(1);

	std::cout << ans;

	return 0;
}