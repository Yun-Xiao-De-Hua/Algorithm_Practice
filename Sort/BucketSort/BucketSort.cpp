#include<vector>
#include<iostream>
constexpr int N = 10010;
int n = 8, w = 49;
int a[N] = {29,25,3,49,9,37,21,43};
std::vector<int> bucket[N];


//void insertionSort(int arr[], int len)
//{
//	for (int i = 1; i < len; i++) {
//		int key = arr[i];
//		int j = i - 1;
//		while (j >= 0 && arr[j] < key) {
//			arr[j + 1] = arr[j];
//			j--;
//		}
//		arr[j + 1] = key;
//	}
//}

void insertionSort(std::vector<int>&arr)
{
	for (int i = 1; i < arr.size(); i++) {
		int key = arr[i];
		int j = i - 1;
		while (j >= 0 && arr[j] > key) {
			arr[j + 1] = arr[j];
			j--;
		}
		arr[j + 1] = key;
	}
}

void bucketSort()
{
	int bucketSize = w / n + 1;

	for (int i = 0; i < n; i++) {
		bucket[i].clear();
	}

	for (int i = 0; i < n; i++) {
		bucket[a[i] / bucketSize].push_back(a[i]);
	}

	int pos = 0;
	for (int i = 0; i < N; i++) {
		insertionSort(bucket[i]);
		for (int j = 0; j < bucket[i].size(); j++) {
			a[pos++] = bucket[i][j];
		}
	}
}

int main()
{
	bucketSort();

	for (int i = 0; i < n; i++)
		std::cout << a[i] << " ";

	return 0;
}
