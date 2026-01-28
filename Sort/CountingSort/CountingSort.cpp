#include<cstring>
#include<iostream>

constexpr int MAXN = 1010;
constexpr int MAXW = 100010;

int cnt[MAXW], b[MAXN];

int* countingSort(int* a, int n, int w)
{
	memset(cnt, 0, sizeof(cnt));
	for (int i = 1; i <= n; i++) cnt[a[i]]++;
	for (int i = 1; i <= w; i++) cnt[i] += cnt[i - 1];
	for (int i = n; i >= 1; i--) b[cnt[a[i]]--] = a[i];

	return b;
}

int main()
{
	const int size = 11;
	int n = size - 1;
	int w = 10;
	int array[size] = { -1,10,8,9,2,3,5,1,4,6,7 };

	std::cout << "original: " << std::endl;
	for (int i = 1; i <= n; ++i) std::cout << array[i] << " ";

	int* sortedArray = countingSort(array, n, w);
	std::cout << std::endl << "sorted: " << std::endl;
	for (int i = 1; i <= n; ++i) std::cout << sortedArray[i] << " ";

	return 0;
}