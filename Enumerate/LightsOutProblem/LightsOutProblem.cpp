#include<cstring>
#include<iostream>

int initialLights[5][6];
int lights[5][6];	// 模拟状态
int presses[5][6];	// 最终的按键结果

void pressButton(int r, int c)
{
	lights[r][c] = 1 - lights[r][c];

	if (r > 0) {
		lights[r - 1][c] = 1 - lights[r - 1][c];
	}
	if (r < 4) {
		lights[r + 1][c] = 1 - lights[r + 1][c];
	}
	if (c > 0) {
		lights[r][c - 1] = 1 - lights[r][c - 1];
	}
	if (c < 5) {
		lights[r][c + 1] = 1 - lights[r][c + 1];
	}
}

int main()
{
	// 初始化灯列的状态
	for (int r = 0; r < 5; r++) {
		for (int c = 0; c < 6; c++)
			std::cin >> initialLights[r][c];
	}

	// 遍历第一行所有可能的按键状态，递推改变后续行状态
	for (int op = 0; op < 64; op++) {
		std::memcpy(lights, initialLights, sizeof(initialLights));	// 重置模拟状态
		std::memset(presses, 0, sizeof(presses));


		// 确定第一行的按键状态
		for (int c = 0; c < 6; c++) {
			if ((op >> c) & 1) {
				pressButton(0, c);
				presses[0][c] = 1;
			}
		}

		// 遍历后续行进行调整
		for (int r = 1; r < 5; r++) {
			for (int c = 0; c < 6; c++) {
				if (lights[r-1][c] == 1) {
					pressButton(r, c);
					presses[r][c] = 1;
				}
			}
		}

		bool success = true;
		// 检查最后一行是否全部熄灭
		for (int c = 0; c < 6; c++) {
			if (lights[4][c] == 1) {
				success = false;
				break;
			}
		}

		if (success) {
			for (int r = 0; r < 5; r++) {
				for (int c = 0; c < 6; c++) {
					std::cout << presses[r][c] << (c == 5 ? "" : " ");
				}
				std::cout << std::endl;
			}
			return 0;
		}
	}

	return 0;
}