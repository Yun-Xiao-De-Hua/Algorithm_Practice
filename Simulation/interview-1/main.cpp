#include<iostream>
#include<algorithm>
#include<vector>
#include<cmath>
using namespace std;

const int N_p = 105;
const int max_it = 50;
const double tol = 1e-4;

struct pos {
    double x, y;
};

vector<vector<int>> process;
vector<pos> center;
int k, n, speed;
pos packet[N_p];
pos start = { 0,0 };

bool comp(const pos& a, const pos& b) {
    double d_a, d_b;
    d_a = a.x * a.x + a.y * a.y;
    d_b = b.x * b.x + b.y * b.y;
    return d_a < d_b;
}

double calc_dis(const pos& a, const pos& b) {
    double sum = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    return sqrt(sum);
}

int get_distr_index(const pos& p) {
    int distr_i = 0;
    double min_dist = calc_dis(p, center[0]);

    for (int i = 1; i < center.size(); i++) {
        double temp = calc_dis(p, center[i]);
        if (temp < min_dist) {
            min_dist = temp;
            distr_i = i;
        }
    }
    return distr_i;
}

double calc_tol(const vector<pos>& a, const vector<pos>& b)
{
    double sum = 0;
    for (int i = 0; i < k; i++) sum += calc_dis(a[i], b[i]);
    return sum;
}

int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);

    cin >> k >> n >> speed;
    for (int i = 1; i <= n; i++) cin >> packet[i].x >> packet[i].y;

    if (n <= k) {
        for (int i = 1; i <= n; i++) center.push_back({ packet[i].x,packet[i].y });
        sort(center.begin(), center.end(), comp);

        double sum = 0;
        sum += calc_dis(start, center[0]);
        for (int i = 1; i < center.size(); i++) sum += calc_dis(center[i], center[i - 1]);
        sum += calc_dis(center[center.size() - 1], start);

        double t_origin = 3600 * sum / speed;
        cout << (int)t_origin;

        return 0;
    }



    vector<pos> temp;
    temp.resize(n);
    for (int i = 0; i < n; i++) {
        temp[i].x = packet[i + 1].x;
        temp[i].y = packet[i + 1].y;
    }
    sort(temp.begin(), temp.end(), comp);
    for (int i = 0; i < k; i++) center.push_back({ temp[i].x,temp[i].y });

    process.resize(k);
    for (int i = 1; i <= max_it; i++) {
        for (int j = 0; j < k; j++) process[j].clear();

        for (int j = 1; j <= n; j++) {
            int distr_i = get_distr_index(packet[j]);
            if (distr_i >= 0 && distr_i < k) process[distr_i].push_back(j);
        }

        vector<pos>new_center;
        for (int j = 0; j < k; j++) {
            double x_sum = 0, y_sum = 0, x, y;
            for (const int& index : process[j]) {
                x_sum += packet[index].x;
                y_sum += packet[index].y;
            }
            if (process[j].size() != 0) {
                x = x_sum / process[j].size();
                y = y_sum / process[j].size();
                new_center.push_back({ x,y });
            }
            else new_center.push_back(center[j]);
        }

        double move_sum = calc_tol(new_center, center);

        center = new_center;

        if (move_sum < tol) break;
    }

    sort(center.begin(), center.end(), comp);

    double sum = 0;
    sum += calc_dis(start, center[0]);
    for (int i = 1; i < center.size(); i++) sum += calc_dis(center[i], center[i - 1]);
    sum += calc_dis(center[center.size() - 1], start);

    double t_origin = 3600 * sum / speed;
    cout << (int)t_origin;

    return 0;
}