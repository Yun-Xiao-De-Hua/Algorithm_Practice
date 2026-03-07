#include<iostream>

int N;

int main()
{
	std::cin >> N;

	int cnt = 0;
	for (int i = 1; i <= N; ++i) {
		int num = i;
		for (int j = 1; num; ++j) {
			if (j % 2 != (num % 10) % 2) break;
			num /= 10;
		}
		if (num == 0) cnt++;
	}

	std::cout << cnt;

	return 0;
}
