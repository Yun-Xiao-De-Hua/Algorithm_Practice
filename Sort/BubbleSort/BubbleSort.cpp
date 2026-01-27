#include<iostream>
#include<utility>

void bubbleSort(int* a, int n)
{
	bool flag = true;
	while (flag) {
		flag = false;
		for (int i = 1; i < n; i++) {
			if (a[i] > a[i + 1]) {
				std::swap(a[i], a[i + 1]);
				flag = true;
			}
		}
	}
}

int main()
{
	const int size = 11;
	int array[size] = { -1,6,2,3,1,4,5,8,7,9,10 };
	std::cout << "original: " << std::endl;
	for (int i = 1; i < size; i++)
		std::cout << array[i] << " ";

	bubbleSort(array, 7);
	
	std::cout << std::endl << "sorted: " << std::endl;
	for (int i = 1; i < size; i++)
		std::cout << array[i] << " ";

	return 0;
}