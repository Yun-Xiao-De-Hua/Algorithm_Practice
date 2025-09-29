#include<iostream>
#include<fstream>
#include<string>
#include<random>
#include<chrono>
#include<vector>
#include<utility>


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
			buffer[j] = generateRandomInt(-1000, 1000);

		// buffer������ͽ���һ��IO����ֹ�ڴ�ľ�
		outFile.write(reinterpret_cast<const char*>(buffer.data()), recordNumInThisRound * sizeof(int));
	}

	outFile.close();

	std::cout << "Successfully write " << recordNum << " records into the test file" << std::endl;
}


// ���ڿ��ŵ������������
int partitionRandomized(std::vector<int>& arr, int low, int high)
{
	static std::mt19937 engine(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	std::uniform_int_distribution<int> distrib(low, high);

	// ȡһ�����λ����Ϊpivot
	int randomIndex = distrib(engine);

	// ��pivot��������������λ�ã��������
	std::swap(arr[randomIndex], arr[high]);

	int lastSmaller = low - 1;

	for (int i = 0; i < high; i++)
		if (arr[i] < arr[high])
			std::swap(arr[++lastSmaller], arr[i]);

	std::swap(arr[lastSmaller + 1], arr[high]);

	return lastSmaller + 1;
}


// �ڲ������㷨���壺��������
void quickSort(std::vector<int>& arr, int low, int high)
{
	if (low < high) {
		int pivot = partitionRandomized(arr, low, high);
		quickSort(arr, low, pivot - 1);
		quickSort(arr, pivot + 1, high);
	}
}


// ˳������
// inputFile: �����ļ�
// CHUNK_SIZE��ģ����ڴ��С
std::vector<std::string> generateRuns(const std::string intputFile, const int CHUNK_SIZE)
{
	// ��¼˳���ļ����ļ��������ں���˳���ļ��Ĵ���鲢����
	std::vector<std::string> runsFiles;

	std::ifstream infile(intputFile, std::ios::binary);

	int count = 0;

	while (!infile.eof()) {
		// ģ���ڴ��С
		std::vector<int> chunk(CHUNK_SIZE);

		infile.read(reinterpret_cast<char*>(chunk.data()), CHUNK_SIZE);

		// ȷ��ʵ�ʶ����������
		std::streamsize elementsRead = infile.gcount() / sizeof(int);
		//std::cout << std::endl << "infile.gcount: " << infile.gcount() << std::endl;
		//std::cout << "elementsRead: " << elementsRead << std::endl;

		// �����������ȣ������ڲ�����
		if (elementsRead == 0) break;
		chunk.resize(elementsRead);

		// ִ���ڲ������㷨������ʹ������ʵ�ֵĿ������� --> ���ո���sort��ֱ��ʹ�ÿ��ŵݹ���ȹ��󣬵���ջ���
		if (!chunk.empty()) {
			//quickSort(chunk, 0, chunk.size() - 1);
			std::sort(chunk.begin(), chunk.end());
		}


		// ����˳���ļ���
		std::string runsFileName = "run_" + std::to_string(++count) + ".tem";

		std::ofstream outFile(runsFileName, std::ios::binary);
		outFile.write(reinterpret_cast<const char*>(chunk.data()), chunk.size() * sizeof(int));
		outFile.close();

		std::cout << "Runfile ( " << runsFileName << " ) has generated" << std::endl;

		// ��¼˳���ļ���
		runsFiles.push_back(runsFileName);
	}

	infile.close();

	return runsFiles;
}


int main()
{
	std::string testFile = "test_file";
	long long recordNum = 100000000;
	const int CHUNK_SIZE = 36 * 1024 * 1024;	// 36MB���պÿ��Եõ�������˳��

	// �����ļ�����
	std::cout << "Start generating testfile... " << std::endl;
	generateTestFile(testFile, recordNum);
	std::cout << "Successfully generate testfile: ";

	//// �����㷨����
	//std::vector<int> testArr = { 2,3,1,4,6,5,9,7,0,8 };

	//std::cout << "origin: ";
	//for (int ele : testArr)
	//	std::cout << ele << " ";

	//quickSort(testArr, 0, testArr.size() - 1);

	//std::cout << std::endl;
	//std::cout << "result: ";
	//for (int ele : testArr)
	//	std::cout << ele << " ";

	// ˳������
	std::cout << "Start generating runsfile... " << std::endl;
	std::vector<std::string> runsFileNames = generateRuns(testFile, CHUNK_SIZE);
	std::cout << "Runs generation process finished" << std::endl << "Successfully generate runsfile: ";
	for (std::string name : runsFileNames)
		std::cout << name << " ";
	std::cout << std::endl;

	// �鲢����




	return 0;
}