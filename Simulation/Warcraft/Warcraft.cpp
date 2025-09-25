#include<iostream>
#include"Game.h"


void loadGame()
{
	int M, N, T;
	int initialHp[5] = {}, initialPower[5] = {};

	std::cin >> M >> N >> T;

	for (int i = 0; i < 5; i++)
		std::cin >> initialHp[i];
	for (int i = 0; i < 5; i++)
		std::cin >> initialPower[i];

	Game game(M, N, T, initialHp, initialPower);

	game.run();
}

int main()
{
	int caseNum;
	std::cin >> caseNum;

	for (int i = 0; i < caseNum; i++) {
		std::cout << "Case:" << i + 1 << std::endl;
		loadGame();
	}

	return 0;
}