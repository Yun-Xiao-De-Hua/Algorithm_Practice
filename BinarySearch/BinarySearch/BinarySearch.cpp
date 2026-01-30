#include<iostream>

int a[1000010];
int n, m;

bool check(int h)
{
	long long sum = 0;
	for (int i = 1; i <= n; i++) sum += (long long)std::max(0, a[i] - h);
	return sum >= m;
}

int find()
{
	int l = 1, r = 1e9 + 1;
	
	while (l + 1 < r) {
		int mid = l + ((r - l) >> 1);

		if (check(mid))
			l = mid;
		else
			r = mid;
	}

	return l;
}

int main()
{
	std::cin >> n >> m;
	for (int i = 1; i <= n; i++) a[i] = i;
	std::cout << find();
	return 0;
}