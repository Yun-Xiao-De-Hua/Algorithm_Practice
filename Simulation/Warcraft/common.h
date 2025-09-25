#pragma once

#include<string>

enum Color { RED, BLUE };
enum WarriorType { DRAGON, NINJA, ICEMAN, LION, WOLF };
enum Flag { NO_FLAG, RED_FLAG, BLUE_FLAG };

extern const std::string WARRIOR_NAMES[];
extern const std::string COLOR_NAMES[];

void printEvent(int time);