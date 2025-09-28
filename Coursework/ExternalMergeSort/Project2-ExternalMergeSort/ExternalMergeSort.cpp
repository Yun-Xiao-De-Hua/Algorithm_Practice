#include<iostream>
#include<fstream>
#include<cstring>
#include<random>
#include<chrono>
#include<vector>


// ���������
int generateRandomInt(int min, int max)
{
	// �������������
	static std::mt19937 engine(std::chrono::high_resolution_clock::now().time_since_epoch().count());

	// �Ż�ָ�������޷�Χ�ڵľ��ȷֲ�
	std::uniform_int_distribution<int> distrib(min, max);

	return distrib(engine);
}


// �ⲿ�����ļ�����
void generateTestFile(const std::string fileName, long long recordNum)
{
	std::ofstream outFile(fileName, std::ios::binary | std::ios::app);

	// ���建����
	const int CHUNK_SIZE = 10000;
	std::vector<int> buffer(CHUNK_SIZE);

	for (long long i = 0; i < recordNum; i+=CHUNK_SIZE) {

		// ����ÿһ��ʵ��д�������
		long long recordNumInThisRound = std::min((long long)CHUNK_SIZE, recordNum - i);

		for (long long j = 0; j < recordNumInThisRound; j++)
			buffer[j] = generateRandomInt(-10000, 10000);

		// buffer������ͽ���һ��IO����ֹ�ڴ�ľ�
		outFile.write(reinterpret_cast<const char*>(buffer.data()), recordNumInThisRound * sizeof(int));
	}

	outFile.close();

	std::cout << "Successfully write " << recordNum << " records into the test file" << std::endl;
}


int main()
{








	return 0;
}