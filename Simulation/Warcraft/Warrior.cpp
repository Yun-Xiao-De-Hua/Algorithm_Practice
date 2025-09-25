#include<iostream>
#include"Warrior.h"

Warrior::Warrior(int id, WarriorType type, Headquarter* hq, int h, int p, int l, Color c)
	:id(id), type(type), hq(hq), hp(h), power(p), location(l), color(c), icemanSteps(0), wolfKills(0) {}

void Warrior::march()
{

}

void Warrior::attack(Warrior& opponent)
{

}

void Warrior::fightBack(Warrior& opponent)
{

}

bool Warrior::isAlive()
{
	return true;	// TODO
}

void Warrior::earnCityEle(int elements)
{

}

void Warrior::getRewardFromHq()
{

}

// 处理特殊武士

// lion
void Warrior::handleLionDeathEffect(Warrior& killer)
{

}

// wolf
void Warrior::handleWolfKillEffect(Warrior& victim)
{

}

// dragon
void Warrior::yell()
{

}


// 辅助输出函数
void Warrior::printMarchEvent(int time)
{

}

void Warrior::printReachHqEvent(int time)
{

}