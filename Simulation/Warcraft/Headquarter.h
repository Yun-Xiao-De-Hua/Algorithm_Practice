#pragma once

#include"common.h"
#include<memory>
#include<vector>


class Warrior;

class Headquarter
{
public:
	Color color;
	int elements;
	int warriorCount;
	int nextWarriorIndex;	// 指向生产序列的下一个索引
	std::vector<WarriorType> productionSeq;
	int initialHp[5];	// 各类武士的初始生命值
	int initialPower[5];	// 各类武士的初始攻击力
	std::vector<std::unique_ptr<Warrior>> warriors;
	int enemyAtHqNum;	// 到达本部的敌人数量
	bool stopped;	// 是否因生命元不足而暂停生产

	Headquarter(Color color, int ele, const std::vector<WarriorType>& seq);

	// 这里为什么接收time ？？？？？？？？？？？？
	void produceWarrior(int time);

	void addElements(int amount);

	bool rewardWarrior(Warrior* w);

	void reportElement(int time);

};