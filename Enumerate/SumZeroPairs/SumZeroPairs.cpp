#include<cstring>
#include<iostream>
constexpr int N = 10000;


int zeroPairSolution(int arr[], int len)
{
	bool met[2 * N + 1];
	std::memset(met, 0, sizeof(met));

	int ans = 0;
	for (int i = 0; i < len; i++) {
		if (met[-arr[i] + N]) ans++;
		met[arr[i] + N] = true;
	}
	return ans;
}


int main()
{
	int arr[10] = { 1,-1,2,-2,3,-3,4,6,8,-8 };
	int arrSize = 10;
	std::cout << "answer: " << zeroPairSolution(arr, arrSize) << std::endl;

	return 0;
}