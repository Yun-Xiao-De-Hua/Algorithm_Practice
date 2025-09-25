#include"Game.h"


Game::Game(int m, int n, int t, const int initialHp[], const int initialPower[])
	:M(m), N(n), T(t), gameOver(false), redHq(std::make_unique<Headquarter>(RED, m, std::vector<WarriorType>{ICEMAN, LION, WOLF, NINJA, DRAGON})), blueHq(std::make_unique<Headquarter>(BLUE, m, std::vector<WarriorType>{LION, DRAGON, NINJA, ICEMAN, WOLF}))
{
	for (int i = 0; i < 5; i++) {
		redHq->initialHp[i] = initialHp[i];
		redHq->initialPower[i] = initialPower[i];

		blueHq->initialHp[i] = initialHp[i];
		blueHq->initialPower[i] = initialPower[i];
	}

	for (int i = 0; i < N; i++) {
		cities.emplace_back(i);
	}
}

void Game::run()
{
	for (int t = 0; t < T; t += 10) {
		int minute = t % 60;
		switch (minute) {
		case 0: processBirth(t); break;
		case 10: processMarch(t); break;
		case 20: processCityProduction(t); break;
		case 30: processElementCollection(t); break;
		case 40: processBattles(t); break;
		case 50: processHqReport(t); break;
		default: break;
		}
		if (gameOver) break;
	}
}

void Game::processBirth(int time)
{

}

void Game::processMarch(int time)
{

}

void Game::processCityProduction(int time)
{

}

void Game::processElementCollection(int time)
{

}

void Game::processBattles(int time)
{

}

void Game::processHqReport(int time)
{

}