#include<iostream>

const int N = 1e6;
long long L[N], R[N], a[N];
int n;

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(nullptr);

	std::cin >> n;
	for (int i = 1; i <= n; ++i) {
		std::cin >> a[i];
		L[i] = std::max(L[i - 1] + a[i], a[i]);
	}

	for (int i = n; i >=1; --i) {
		R[i] = std::max(R[i + 1] + a[i], a[i]);
	}

	for (int i = 1; i <= n; ++i) {
		std::cout << L[i] + R[i] - a[i] << ' ';
	}

	return 0;
}