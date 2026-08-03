#include<iostream>
#include<queue>
#include<cstring>
using namespace std;

const int N = 410;
int dist[N + 5][N + 5];

struct pos {
	int x, y;
};

int dx[9] = {0,-2,-2,-1,1,2,2,-1,1};
int dy[9] = {0,-1,1,2,2,-1,1,-2,-2};

void bfs_all(int n, int m, pos s) 
{
	memset(dist, -1, sizeof(dist));

	queue<pos> q;

	dist[s.x][s.y] = 0;
	q.push(s);

	while (!q.empty()) {
		pos p = q.front();
		q.pop();

		for (int i = 1; i <= 8; ++i) {
			int nx = p.x + dx[i];
			int ny = p.y + dy[i];

			if (nx >= 1 && nx <= n && ny >= 1 && ny <= m && dist[nx][ny] == -1) {
				dist[nx][ny] = dist[p.x][p.y] + 1;
				q.push({ nx,ny });
			}
		}
	}
}

int main()
{
	ios::sync_with_stdio(0);
	cin.tie(0);

	int n, m, x, y; cin >> n >> m >> x >> y;

	bfs_all(n, m, { x,y });

	for (int i = 1; i <= n; ++i) {
		for (int j = 1; j <= m; ++j) {
			cout << dist[i][j] << ' ';
		}
		cout << '\n';
	}

	return 0;
}