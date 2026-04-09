#include<iostream>
#include<algorithm>
#include<vector>
using namespace std;

const int N = 1e3 + 5;
int a[N], b[N], c[N], n, m, sum;
int ap, bp, cp;

int game(int x, int y, int z) {
	int maxn = 0;
	if (x == y && y == z) maxn = max(maxn, 200);

	if(x==y||x==z||y==z) maxn = max(maxn, 100);

	if(y-x==1&&z-y==1) maxn = max(maxn, 200);

	vector<int> temp = { x,y,z };
	sort(temp.begin(), temp.end());
	if(temp[2]-temp[1]==1&&temp[1]-temp[0]==1) maxn = max(maxn, 100);

	return maxn;
}

int main()
{
	ios::sync_with_stdio(0);
	cin.tie(0);

	cin >> n;
	for (int i = 0; i < n; i++) cin >> a[i];
	for (int i = 0; i < n; i++) cin >> b[i];
	for (int i = 0; i < n; i++) cin >> c[i];
	cin >> m;
	
	while (m--) {
		int x, y, z;
		cin >> x >> y >> z;

		ap = (ap + x) % n;
		bp = (bp + y) % n;
		cp = (cp + z) % n;

		sum += game(a[ap], b[bp], c[cp]);
	}

	cout << sum;

	return 0;
}