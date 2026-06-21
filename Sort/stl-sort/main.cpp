#include<iostream>
#include<algorithm>

const int N = 2e5 + 10;

struct book {
	int h, s, w;
}b[N];

bool cmp(const book& b1, const book& b2){
	if (b1.h == b2.h && b1.s == b2.s) return b1.w > b2.w;
	else if (b1.h == b2.h) return b1.s > b2.s;
	else return b1.h > b2.h;
}

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	int n; std::cin >> n;

	for (int i = 1; i <= n; i++) std::cin >> b[i].h >> b[i].s >> b[i].w;

	std::sort(b + 1, b + 1 + n, cmp);

	for (int i = 1; i <= n; i++) std::cout << b[i].h << ' ' << b[i].s << ' ' << b[i].w << '\n';

	return 0;
}