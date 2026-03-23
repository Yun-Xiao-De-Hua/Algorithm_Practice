// 哈希表的使用，平均检索时间达O(1)
// 结合强类型枚举和Switch进行有意义的分支

#include<iostream>
#include<unordered_map>
#include<string>

std::string n;
int s;
std::unordered_map<std::string, int> m;
int Q, d;

enum class Operation {
	INSERTORMODIFY = 1,
	SEARCH,
	DELETE,
	PRINT
};

bool try_parse_input(int input, Operation& output)
{
	if (input >= 1 && input <= 4) {
		output = static_cast<Operation>(input);
		return true;
	}
	return false;
}

void process(int input) {
	Operation op;
	if (!try_parse_input(input, op)) {
		std::cerr << "Invalid input" << '\n';
		return;
	}

	switch (op) {
	case Operation::INSERTORMODIFY:
		std::cin >> n >> s;
		m[n] = s;
		std::cout << "OK" << '\n';
		break;
	case Operation::SEARCH:
		std::cin >> n;
		if (m.find(n) != m.end()) std::cout << m[n] << '\n';
		else std::cout << "Not found" << '\n';
		break;
	case Operation::DELETE: {
		std::cin >> n;
		auto it = m.find(n);
		if (it == m.end()) std::cout << "Not found" << '\n';
		else {
			m.erase(it);
			std::cout << "Deleted successfully" << '\n';
		}
		break;
	}
	case Operation::PRINT:
		std::cout << m.size() << '\n';
		break;
	default:
		std::cerr << "Something wrong" << '\n';
	}
}

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);

	std::cin >> Q;

	while (Q--) {
		std::cin >> d;
		process(d);
	}

	return 0;
}