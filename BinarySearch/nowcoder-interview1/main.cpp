#include <iostream>
#include <climits>
#include <algorithm>
using namespace std;

const int N = 1e3 + 5;
int t, k, a[N], close[N];
int minp = INT_MAX;

bool check(int dep)
{
    bool pass = true;
    int last_pos = 0;
    for (int i = 1; i <= dep; i++) {
        if (last_pos) {
            if (close[last_pos] == 1 && close[i] == 1) {
                pass = false;
                break;
            }
        }
        last_pos = i;
    }

    return pass;
}

void dfs(int dep, int has_close, int d)
{
    if (!check(dep)) return;

    if (dep > t) {
        if (has_close == k) {
            minp = min(minp, d);
        }
        return;
    }

    if (dep != t) {
        close[dep] = 1;
        dfs(dep + 1, has_close + 1, d + a[dep]);
        close[dep] = 0;
    }

    dfs(dep + 1, has_close, d);
}

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);

    cin >> t >> k;
    for (int i = 1; i <= t - 1; i++) cin >> a[i];

    dfs(1, 0, 0);

    if (minp == INT_MAX) cout << -1;
    else cout << minp;

    return 0;
}