#include<iostream>

int array[5][5];
int m, n;

int main()
{
	for (int i = 0; i < 5; ++i) {
		for (int j = 0; j < 5; ++j) {
			std::cin >> array[i][j];
		}
	}

	std::cin >> m >> n;

	for (int i = 0; i < 5; ++i) {
		if (i == m - 1) {
			for (int j = 0; j < 5; ++j) {
				std::cout << array[n - 1][j] << ' ';
			}
		}
		else if (i == n - 1) {
			for (int j = 0; j < 5; ++j) {
				std::cout << array[m - 1][j] << ' ';
			}
		}
		else {
			for (int j = 0; j < 5; ++j) {
				std::cout << array[i][j] << ' ';
			}
		}

		std::cout << std::endl;
	}

	return 0;
}