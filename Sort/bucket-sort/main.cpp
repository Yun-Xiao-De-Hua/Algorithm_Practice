// 给定序列值域范围小，但序列本身长度较大，考虑使用桶排序

#include<iostream>

const int R = 2e5 + 10;
int bucket[R];

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	int n; std::cin >> n;

	for (int i = 1; i <= n; i++) {
		int num; std::cin >> num;
		bucket[num]++;
	}

	for (int i = 0; i <= 2e5; i++)
		for (int j = 1; j <= bucket[i]; j++) std::cout << i << ' ';

	return 0;
}