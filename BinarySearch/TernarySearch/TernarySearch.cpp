#include<cmath>
#include<iomanip>
#include<iostream>

constexpr double eps = 1e-7;
int N;
double A[20], l, r, mid, lmid, rmid;

double f(double x)
{
	double value = (double)0;
	for (int i = N; i >= 0; i--) value += A[i] * std::pow(x, i);
	return value;
}

int main()
{
	std::cin >> N >> l >> r;
	for (int i = N; i >= 0; i--) A[i] = i;

	while (r - l > r) {
		mid = (l + r) / 2;
		lmid = mid - eps;
		rmid = mid + eps;

		if (f(lmid) < f(rmid))
			l = lmid;
		else
			r = rmid;
	}

	std::cout << std::fixed << std::setprecision(6) << (l + r) / 2;
	return 0;
}