/*#include<iostream>

int p[1000010];
int c[1000010];

int n;
long long S;

int main()
{
	std::cin >> n >> S;

	int max_train = 1;
	for (int i = 1; i <= n; ++i) {
		std::cin >> p[i] >> c[i];
		if (c[i] > max_train) max_train = c[i];
	}

	long long cost = 0;
	for (int i = 0; i <= max_train; ++i) {
		long long sum = i * S;
		for (int j = 1; j <= n; ++j) {
			int rem_train = std::max(0, c[j] - i);
			sum += (long long)p[j] * rem_train;
		}
		
		if (cost == 0 || sum < cost) cost = sum;
	}

	std::cout << cost;

	return 0;
}*/

#include<iostream>
typedef long long LL;
const int N = 1e6 + 10;
LL n, S, p[N], c[N], bucket[N], sum, epoch, group;

int main()
{
	std::cin >> n >> S;
	for (int i = 1; i <= n; ++i) {
		std::cin >> p[i] >> c[i];
		bucket[c[i]] += p[i];
		epoch += p[i];
		sum += p[i] * c[i];
	}

	for (int i = 1; i <= 1e6; ++i) {
		if (epoch < S) break;
		group += S;
		sum -= epoch;
		epoch -= bucket[i];
	}

	std::cout << group + sum;

	return 0;
}