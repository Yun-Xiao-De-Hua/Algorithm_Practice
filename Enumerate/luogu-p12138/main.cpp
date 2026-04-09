#include<iostream>
#include<cmath>

using namespace std;

bool check(int n) {
	bool pass = true;
	for (int i = 2; i <= sqrt(n); i++) {
		if (n % i == 0) {
			pass = false;
			break;
		}
	}

	return pass;
}

int main()
{
	int ans;
	int cnt = 0 , i =2;
	while (cnt < 2025) {
		if (check(i)) cnt++;
		if (cnt == 2025) ans = i;
		i++;
	}

	cout << ans;

	return 0;
}