#include<iostream>
#include<algorithm>

void insertionSort(int arr[], int len)
{
	for (int i = 1; i < len; i++) {
		int key = arr[i];
		int j = i - 1;
		while (j >= 0 && arr[j] < key) {
			arr[j + 1] = arr[j];
			j--;
		}
		arr[j + 1] = key;
	}
}

void insertionSortByRight(int arr[], int len)
{
	for (int i = len - 2; i >= 0; i--) {
		int key = arr[i];
		int j = i + 1;
		while (j < len && arr[j] < key) {
			arr[j - 1] = arr[j];
			j++;
		}
		arr[j - 1] = key;
	}
}

void binaryInsertionSort(int arr[], int len)
{
	if (len < 2)
		return;
	for (int i = 1; i < len; i++) {
		int key = arr[i];
		int index = std::upper_bound(arr, arr + i, key) - arr;
		memmove(arr + i + 1, arr + i, (i - index) * sizeof(int));
		arr[index] = key;
	}
}




int main() {
	const int len = 5;
	int testArr[len] = {3,2,1,5,4};
	//insertionSort(testArr, len);
	insertionSortByRight(testArr, len);

	for (int i = 0; i < len; i++) {
		std::cout << testArr[i] << " ";
	}

	return 0;
}