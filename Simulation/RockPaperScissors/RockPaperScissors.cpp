#include<iostream>

constexpr int supper = 210;
int gamerA[supper];
int gamerB[supper];

int rules[5][5] = {
	{0,-1,1,1,-1},
	{1,0,-1,1,-1},
	{-1,1,0,-1,1},
	{-1,-1,1,0,1},
	{1,1,-1,-1,0}
};

int getGameResult(int gamerA, int gamerB)
{
	return rules[gamerA][gamerB];
}

std::pair<int, int> simulation(int N, int Na, int Nb, int cycleA[], int cycleB[])
{
	int resultA = 0, resultB = 0;

	for (int i = 0; i < N; i++) {
		int choiceA = cycleA[i % Na];
		int choiceB = cycleB[i % Nb];

		int gameResult = getGameResult(choiceA, choiceB);

		if (gameResult == 1) resultA++;
		else if (gameResult == -1) resultB++;
	}

	return std::pair<int, int>(resultA, resultB);
}


int main()
{
	int N, Na, Nb;
	std::cin >> N >> Na >> Nb;

	for (int i = 0; i < Na; i++)
		std::cin >> gamerA[i];

	for (int i = 0; i < Nb; i++)
		std::cin >> gamerB[i];

	std::pair<int, int>result = simulation(N, Na, Nb, gamerA, gamerB);

	std:: cout << result.first << " " << result.second;

	return 0;
}