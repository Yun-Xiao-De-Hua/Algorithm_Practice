// STL的使用，利用set的平衡性，结合lower_bound进行二分搜索

#include<iostream>
#include<set>

int m, d, l;
std::set<int> s;

enum class Operation {
	IMPORT = 1,
	EXPORT
};

bool try_parse_enter(int num, Operation& output)
{
	if (num >= 1 && num <= 2) {
		output = static_cast<Operation>(num);
		return true;
	}
	return false;
}

void process(int enter)
{
	Operation op;
	if (!try_parse_enter(enter, op)) {
		std::cerr << "Invalid input" << '\n';
		return;
	}

	switch (op) {
	case Operation::IMPORT:
		if (s.find(l) != s.end()) std::cout << "Already Exist" << '\n';
		else s.insert(l);
		break;
	case Operation::EXPORT: {
		if (s.empty()) {
			std::cout << "Empty" << '\n';
			break;
		}

		auto it = s.lower_bound(l);
		if (it != s.end() && *it == l) {
			std::cout << *it << '\n';
			s.erase(it);
		}
		else {
			if (it == s.begin()) {
				std::cout << *it << '\n';
				s.erase(it);
			}
			else if (it == s.end()) {
				std::cout << *(--it) << '\n';
				s.erase(it);
			}
			else {
				auto it_prev = std::prev(it);
				auto it_next = it;
				if (l - *it_prev <= *it_next - l) {
					std::cout << *it_prev << '\n';
					s.erase(it_prev);
				}
				else {
					std::cout << *it_next << '\n';
					s.erase(it_next);
				}
			}
		}
		break;
	}
	default:
		std::cerr << "Something wrong" << '\n';
	}
}

int main()
{
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);

	std::cin >> m;
	while (m--) {
		std::cin >> d >> l;
		process(d);
	}

	return 0;
}