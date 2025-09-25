#pragma once

#include"common.h"

class Headquarter;

class Warrior
{
public:
	int id;
	int hp;
	int power;
	int location;
	WarriorType type;
	Color color;
	Headquarter* hq;
	int icemanSteps;
	int wolfKills;

	Warrior(int id, WarriorType type, Headquarter* hq, int h, int p, int l, Color c);

	void march();

	void attack(Warrior& opponent);

	void fightBack(Warrior& opponent);

	bool isAlive();

	void earnCityEle(int elements);

	void getRewardFromHq();


	// ����������ʿ

	// lion
	void handleLionDeathEffect(Warrior& killer);

	// wolf
	void handleWolfKillEffect(Warrior& victim);

	// dragon
	void yell();

	// �����������
	void printMarchEvent(int time);

	void printReachHqEvent(int time);
};