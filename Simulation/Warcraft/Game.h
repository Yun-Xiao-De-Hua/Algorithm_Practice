#pragma once

#include<vector>
#include<memory>
#include"common.h"
#include"Warrior.h"
#include"Headquarter.h"
#include"City.h"


class Game
{
public:
	int M, N, T;
	std::unique_ptr<Headquarter> redHq, blueHq;
	std::vector<City> cities;
	bool gameOver;

	Game(int m, int n, int t, const int initialHp[], const int initialPower[]);

	void run();

	void processBirth(int time);

	void processMarch(int time);

	void processCityProduction(int time);

	void processElementCollection(int time);

	void processBattles(int time);

	void processHqReport(int time);
};