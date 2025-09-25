#include"common.h"
#include<iostream>
#include<iomanip>



const std::string WARRIOR_NAMES[] = { "dragon", "ninja", "iceman", "lion", "wolf" };
const std::string COLOR_NAMES[] = { "red", "blue" };

void printEvent(int time)
{
	std::cout << std::setw(3) << std::setfill('0') << time / 60 << ":" << std::setw(2) << std::setfill('0') << time % 60;
}