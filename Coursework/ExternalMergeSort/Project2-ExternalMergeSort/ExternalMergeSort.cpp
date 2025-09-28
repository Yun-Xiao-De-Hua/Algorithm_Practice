#include<iostream>
#include<fstream>
#include<cstring>
#include<random>
#include<chrono>
#include<vector>


// 随机数生成
int generateRandomInt(int min, int max)
{
	// 定义随机数引擎
	static std::mt19937 engine(std::chrono::high_resolution_clock::now().time_since_epoch().count());

	// 放回指定上下限范围内的均匀分布
	std::uniform_int_distribution<int> distrib(min, max);

	return distrib(engine);
}


// 外部测试文件生成
void generateTestFile(const std::string fileName, long long recordNum)
{
	std::ofstream outFile(fileName, std::ios::binary | std::ios::app);

	// 定义缓存区
	const int CHUNK_SIZE = 10000;
	std::vector<int> buffer(CHUNK_SIZE);

	for (long long i = 0; i < recordNum; i+=CHUNK_SIZE) {

		// 计算每一次实际写入的数量
		long long recordNumInThisRound = std::min((long long)CHUNK_SIZE, recordNum - i);

		for (long long j = 0; j < recordNumInThisRound; j++)
			buffer[j] = generateRandomInt(-10000, 10000);

		// buffer存满后就进行一次IO，防止内存耗尽
		outFile.write(reinterpret_cast<const char*>(buffer.data()), recordNumInThisRound * sizeof(int));
	}

	outFile.close();

	std::cout << "Successfully write " << recordNum << " records into the test file" << std::endl;
}


int main()
{








	return 0;
}