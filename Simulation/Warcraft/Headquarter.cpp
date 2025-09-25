#include<iostream>
#include"Headquarter.h"
#include"Warrior.h"


Headquarter::Headquarter(Color color, int ele, const std::vector<WarriorType>& seq) :warriorCount(0), nextWarriorIndex(0), enemyAtHqNum(0), stopped(false)
{
	warriors.clear();
}

// 这里为什么接收time ？？？？？？？？？？？？
void Headquarter::produceWarrior(int time)
{

}

void Headquarter::addElements(int amount)
{

}

bool Headquarter::rewardWarrior(Warrior* w)
{
	return true;	// TODO
}

void Headquarter::reportElement(int time)
{

}