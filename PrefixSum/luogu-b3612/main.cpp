#include<iostream>

int a[100010],l[100010],r[100010],s[100010];
int n, m;

int main()
{
	std::cin >> n;
	for (int i = 1; i <= n; ++i) {
		std::cin >> a[i];
		s[i] += (a[i] + s[i - 1]);
	}
	std::cin >> m;
	for (int i = 1; i <= m; ++i) {
		std::cin >> l[i] >> r[i];
	}

	for (int i = 1; i <= m; ++i) {
		std::cout << s[r[i]] - s[l[i]-1] << std::endl;
	}

	return 0;
}