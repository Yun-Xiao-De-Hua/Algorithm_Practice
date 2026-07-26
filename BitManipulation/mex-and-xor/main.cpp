#include<iostream>

int getDiff(int n)
{
	int rem = n % 4;

	if (rem == 0) return n;
	else if (rem == 1) return 1;
	else if (rem == 2) return n + 1;
	else return 0;
}

int solve(int a, int b) 
{
	int diff = getDiff(a - 1);
	int check = diff ^ b;
	if (check == 0) return a;
	else if (check == a) return a + 2;
	else return a + 1;
}

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	int t; std::cin >> t;

	while (t--)
	{
		int a, b; std::cin >> a >> b;
		std::cout<< solve(a, b) << '\n';
	}

	return 0;
}