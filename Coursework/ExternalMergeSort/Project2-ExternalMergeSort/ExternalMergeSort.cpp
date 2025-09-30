#include<iostream>
#include<fstream>
#include<string>
#include<random>
#include<chrono>
#include<vector>
#include<utility>
#include<cstdio>


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


// ����������ģ��ʹ���������뻺������һ������������鲢�����ļ�
void mergeTwoFiles(std::string& inputFile1, std::string& inputFile2, std::string& outputFile)
{
	// ���ļ�
	std::ifstream in1(inputFile1, std::ios::binary);
	std::ifstream in2(inputFile2, std::ios::binary);
	std::ofstream out(outputFile, std::ios::binary | std::ios::app);

	// ��������/�����������С
	const int BUFFER_SIZE = 1024;	// 4KB/sizeof(int)

	// ��������/���������
	std::vector<int> buffer1(BUFFER_SIZE);
	std::vector<int> buffer2(BUFFER_SIZE);
	std::vector<int> bufferOut(BUFFER_SIZE);

	// ����������ָ��Ӧ�÷��ʵ�λ��
	size_t idx1 = 0, idx2 = 0, idxOut = 0;

	// ������뻺����
	in1.read(reinterpret_cast<char*>(buffer1.data()), BUFFER_SIZE * sizeof(int));
	size_t size1 = in1.gcount() / sizeof(int);

	in2.read(reinterpret_cast<char*>(buffer2.data()), BUFFER_SIZE * sizeof(int));
	size_t size2 = in2.gcount() / sizeof(int);

	// ֻҪ���뻺�����ǿգ��ͼ����鲢
	while (idx1 < size1 || idx2 < size2) {
		// ������������������һ��IO
		if (idxOut == BUFFER_SIZE) {
			out.write(reinterpret_cast<const char*>(bufferOut.data()), BUFFER_SIZE * sizeof(int));
			idxOut = 0;
		}

		// �����뻺����2�ľ���ֻ�ܴӻ�����1����
		// ��������1Ԫ��С�ڻ�����2Ԫ�أ��ӻ�����1����
		if (idx2 >= size2 || (idx1 < size1 && buffer1[idx1] < buffer2[idx2])) {
			bufferOut[idxOut++] = buffer1[idx1++];

			// �������뻺����
			if (idx1 >= size1) {
				in1.read(reinterpret_cast<char*>(buffer1.data()), BUFFER_SIZE * sizeof(int));
				size1 = in1.gcount() / sizeof(int);
				idx1 = 0;
			}
		}
		else {
			bufferOut[idxOut++] = buffer2[idx2++];

			// �������뻺����
			if (idx2 >= size2) {
				in2.read(reinterpret_cast<char*>(buffer2.data()), BUFFER_SIZE * sizeof(int));
				size2 = in2.gcount() / sizeof(int);
				idx2 = 0;
			}
		}
	}

	// �����������δд���ʣ������д������ļ�
	if (idxOut > 0)
		out.write(reinterpret_cast<const char*>(bufferOut.data()), idxOut * sizeof(int));

	// �ر��ļ�
	in1.close();
	in2.close();
	out.close();
}

// ˳���鲢����
void mergeRuns(std::vector<std::string>& runFiles, const std::string& finalResultFile)
{
	// ��¼�鲢������������ʱ˳���ļ�����
	int passCount = 0;

	// ֱ���鲢��һ���ļ�
	while (runFiles.size() > 1) {
		std::vector<std::string> nextPassRunFiles;

		for (int i = 0; i < runFiles.size(); i += 2) {
			// ż�������鲢�ļ�
			if (i + 1 < runFiles.size()) {
				std::string mergedFileName = "pass_" + std::to_string(++passCount) + "_merge_" + std::to_string(i / 2) + ".tem";
				mergeTwoFiles(runFiles[i], runFiles[i + 1], mergedFileName);
				nextPassRunFiles.push_back(mergedFileName);
			}
			// ���������鲢�ļ���ֱ�ӽ�����һ�ֹ鲢
			else {
				nextPassRunFiles.push_back(runFiles[i]);
			}
		}

		// ������ʱ˳���ļ�
		for (const std::string& runName : runFiles) {
			std::remove(runName.c_str());
		}

		// ����˳���ļ�
		runFiles = nextPassRunFiles;
	}

	// ���������յĽ���ļ�
	if (!runFiles.empty())
		std::rename(runFiles[0].c_str(), finalResultFile.c_str());
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

	/*
	// �����㷨����
	std::vector<int> testArr = { 2,3,1,4,6,5,9,7,0,8 };

	std::cout << "origin: ";
	for (int ele : testArr)
		std::cout << ele << " ";

	quickSort(testArr, 0, testArr.size() - 1);

	std::cout << std::endl;
	std::cout << "result: ";
	for (int ele : testArr)
		std::cout << ele << " ";
	*/

	// ˳������
	std::cout << "Start generating runsfile... " << std::endl;
	std::vector<std::string> runsFileNames = generateRuns(testFile, CHUNK_SIZE);
	std::cout << "Runs generation process finished" << std::endl << "Successfully generate runsfile: ";
	for (std::string name : runsFileNames)
		std::cout << name << " ";
	std::cout << std::endl;

	// �鲢����
	const std::string resultFileName = "sort_output.bin";
	mergeRuns(runsFileNames, resultFileName);

	// ��鲿�ֹ鲢�ļ����
	const int CHECK_SIZE = 100000;
	std::vector<int> checkArr(CHECK_SIZE);

	std::ifstream in(resultFileName, std::ios::binary);
	in.read(reinterpret_cast<char*>(checkArr.data()), CHECK_SIZE * sizeof(int));
	int realSize = in.gcount() / sizeof(int);

	std::cout << "Final result: " << std::endl;
	for (int i = 0; i < realSize;i ++) {
		std::cout << checkArr[i] << ", ";
		if (i > 0 && i % 10 == 0)
			std::cout << std::endl;
	}
	return 0;
}