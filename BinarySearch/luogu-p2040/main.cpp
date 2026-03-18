#include<iostream>
#include<string>
#include<cstring>

//action: 110100000, 111010000, 011001000, 100110100, 010111010, 001011001, 000100110, 000010111, 000001011
std::string s[10] = { "0", "110100000", "111010000","011001000","100110100","010111010","001011001","000100110","000010111","000001011" };
int a[10];

int f[512], q[515], l, r;

void ini() {
	for (int i = 1; i <= 9; i++) {
		int val = 0;
		for (int j = 0; j < 9; j++)
			val += (s[i][j] - '0') * (1 << (8 - j));
		a[i] = val;
		//std::cout << a[i] << ' ';
	}
}

void bfs(int x)
{
	f[x] = 0;
	q[++r] = x;

	while (l < r) {
		int cur = q[++l];
		for (int i = 1; i <= 9; i++) {
			int news = cur ^ a[i];
			if (f[news] == -1) {
				q[++r] = news;
				f[news] = f[cur] + 1;
			}
		}
	}
}

int main() {
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);

	ini();
	std::memset(f, -1, sizeof(f));

	int x = 0;
	for (int i = 8; i >= 0; i--) {
		int n;
		std::cin >> n;
		x += n * (1 << i);
	}

	bfs(x);

	std::cout << f[511];

	return 0;
}