#include<iostream>
#include<cmath>
#include<vector>
#include<algorithm>
#include<iomanip>
using namespace std;

#define Pi acos(-1)

int n;
double x, y, r, risk_sum, line;
struct section {
	double l, r;
};
vector<section> risk;

bool comp(const section& a, const section& b){
	return a.l < b.l;
}

int main()
{
	ios::sync_with_stdio(0);
	cin.tie(0);

	cin >> n;
	while (n--) {
		cin >> x >> y >> r;
		double edge = sqrt(x * x + y * y);
		double alpha = atan(y / x);
		double beta = asin(r / edge);
		risk.push_back({ alpha - beta,alpha + beta });
	}

	sort(risk.begin(), risk.end(), comp);

	for (const auto& sct : risk) {
		line = max(line, sct.l);
		if (line < sct.r) risk_sum += sct.r - line;
		line = max(line, sct.r);
	}

	cout << fixed << setprecision(3) << (Pi / 2 - risk_sum) / (Pi / 2);

	return 0;
}