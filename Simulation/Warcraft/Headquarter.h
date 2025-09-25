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
	int nextWarriorIndex;	// ָ���������е���һ������
	std::vector<WarriorType> productionSeq;
	int initialHp[5];	// ������ʿ�ĳ�ʼ����ֵ
	int initialPower[5];	// ������ʿ�ĳ�ʼ������
	std::vector<std::unique_ptr<Warrior>> warriors;
	int enemyAtHqNum;	// ���ﱾ���ĵ�������
	bool stopped;	// �Ƿ�������Ԫ�������ͣ����

	Headquarter(Color color, int ele, const std::vector<WarriorType>& seq);

	// ����Ϊʲô����time ������������������������
	void produceWarrior(int time);

	void addElements(int amount);

	bool rewardWarrior(Warrior* w);

	void reportElement(int time);

};