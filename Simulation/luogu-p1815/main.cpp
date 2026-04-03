// 模拟问题
// 每次移动只需检查蠕虫头部状态即可，并且蠕虫只可能处于碰撞自己或出界其中一种状态

#include<iostream>
#include<string>

int grid[52][52], n, m, error;
struct node { 
	int x, y; 
	bool operator==(const node& other) const {
		return x == other.x && y == other.y;
	}
	void operator=(const node& other) {
		x = other.x, y = other.y;
	}
} worm[21];
int x[5] = { 0,0,1,0,-1 };
int y[5] = { 0,1,0,-1,0 };

void move(int dir)
{
	for (int i = 1; i < 20; i++) worm[i] = worm[i + 1];
	worm[20].x += x[dir];
	worm[20].y += y[dir];

	for (int i = 1; i < 20; i++) {
		if (worm[20] == worm[i]) {
			error = 1;
			return;
		}
	}

	if (worm[20].x > 50 || worm[20].x < 1 || worm[20].y>50 || worm[20].y < 1) {
		error = 2;
		return;
	}

	error = 0;
}

int main()
{
	std::ios::sync_with_stdio(0);
	std::cin.tie(0);

	while (std::cin >> n && n) {
		std::string input;
		std::cin >> input;

		for (int i = 1; i <= 20; i++) {
			worm[i].x = 25;
			worm[i].y = i + 10;
		}

		for (int i = 0; i < n; i++) {
			if (input[i] == 'E') move(1);
			else if(input[i] == 'S') move(2);
			else if(input[i] == 'W') move(3);
			else if(input[i] == 'N') move(4);

			m = i + 1;
			if (error) break;			
		}

		if (error == 0) std::cout << "The worm successfully made all " << m << " moves.\n";
		else if (error == 1) std::cout << "The worm ran into itself on move " << m << ".\n";
		else if (error == 2) std::cout << "The worm ran off the board on move " << m << ".\n";
	}

	return 0;
}