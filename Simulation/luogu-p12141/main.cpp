#include<iostream>

using namespace std;

const int N = 20;
const int SIZE = (1 << 20);
int t[SIZE], m;

int main()
{
	ios::sync_with_stdio(0);
	cin.tie(0);

	cin >> m;
	
	for (int i = 1; i <= SIZE -(1<<(N-1)); i++) {
		if (t[i] == 0) t[2 * i + 1] = 1;
		if (t[i] == 1) t[2 * i] = 1;
	}

	while (m--) {
		int r, c;
		cin >> r >> c;
		int idx = ((1 << (r - 1)) + c - 1);
		if (t[idx]) cout << "BLACK\n";
		else cout << "RED\n";
	}

	return 0;
}