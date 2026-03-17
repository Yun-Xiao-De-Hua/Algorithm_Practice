//#include<iostream>
//
//int N, M, w = 2, n, m, ans;
//
//void dfs(int dep){
//	if (dep == M + N + 1) {
//		ans++;
//		return;
//	}
//
//	if (m < M) {
//		if (!w) return;
//		if (dep == N + M && w - 1 != 0) return;
//		m++;
//		w -= 1;
//		dfs(dep + 1);
//		m--;
//		w += 1;
//	}
//
//	if (n < N) {
//		if (dep == N + M) return;
//		n++;
//		w *= 2;
//		dfs(dep + 1);
//		n--;
//		w /= 2;
//	}
//}
//
//int main()
//{
//	std::ios::sync_with_stdio(false);
//	std::cin.tie(0);
//
//	std::cin >> N >> M;
//
//	dfs(1);
//
//	std::cout << ans % (int)(1e9 + 7);
//
//	return 0;
//}

#include<iostream>

int N, M, p[210][105][105];

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);

	std::cin >> N >> M;

	p[0][0][2] = 1;
	for (int i = 0; i < N + M; i++) {
		for (int j = 0; j < M; j++) {
			for (int k = 0; k <= M; k++) {
				if (p[i][j][k]) {
					if (k > 0) 
						p[i + 1][j + 1][k - 1] = (p[i + 1][j + 1][k - 1] + p[i][j][k]) % (int)(1e9 + 7);
					if (k <= 50)
						p[i + 1][j][k * 2] = (p[i + 1][j][k * 2] + p[i][j][k]) % (int)(1e9 + 7);
				}
			}
		}
	}

	std::cout << p[N + M - 1][M - 1][1];

	return 0;
}