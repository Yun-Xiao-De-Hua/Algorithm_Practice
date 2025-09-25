#pragma once

#include"common.h"

class Warrior;


class City
{
public:
	int id;
	int element;
	Warrior* redWarrior;
	Warrior* blueWarrior;
	Flag flag;
	Color lastBattleWinner;
	int consecutiveWinsNum;	// ͬһ��Ӫ��ʤ����

	City(int id);

	void produceElements();

	void warriorCollctEle();

	void performBattle(int time);
};