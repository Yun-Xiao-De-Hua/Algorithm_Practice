#include<iostream>
#include<algorithm>
#include<functional>
#include<random>
#include<iomanip>

constexpr int N = 100010;
constexpr double eps = 1e-6;

int n, k;
int a[N], b[N];
double c[N];

bool check(double mid)
{
	double sum = (double)0;
	for (int i = 1; i <= n; ++i) c[i] = a[i] - b[i] * mid;
	std::sort(c + 1, c + 1 + n, std::greater<double>());
	for (int i = 1; i <= k; ++i) sum += c[i];
	return sum >= 0;
}

int main()
{
	std::cin >> n >> k;
	unsigned int seed = 42;
	std::mt19937 gen(seed);
	std::uniform_int_distribution<> dis(0, 1000);
	for (int i = 1; i <= n; ++i) {
		a[i] = dis(gen);
		b[i] = dis(gen);
	}

	double l = 0, r = 1e9;
	while (r - l > eps) {
		double mid = (l + r) / 2;
		if (check(mid))
			l = mid;
		else
			r = mid;
	}
	std::cout << std::fixed << std::setprecision(6) << l << std::endl;

	return 0;
}