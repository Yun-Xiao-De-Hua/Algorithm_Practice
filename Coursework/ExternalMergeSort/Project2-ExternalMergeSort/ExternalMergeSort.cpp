#include<iostream>
#include<fstream>
#include<string>
#include<random>
#include<chrono>
#include<vector>
#include<utility>
#include<cstdio>


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
			buffer[j] = generateRandomInt(-1000, 1000);

		// buffer存满后就进行一次IO，防止内存耗尽
		outFile.write(reinterpret_cast<const char*>(buffer.data()), recordNumInThisRound * sizeof(int));
	}

	outFile.close();

	std::cout << "Successfully write " << recordNum << " records into the test file" << std::endl;
}


// 用于快排的随机分区函数
int partitionRandomized(std::vector<int>& arr, int low, int high)
{
	static std::mt19937 engine(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	std::uniform_int_distribution<int> distrib(low, high);

	// 取一个随机位置作为pivot
	int randomIndex = distrib(engine);

	// 将pivot交换到数组最后的位置，方便遍历
	std::swap(arr[randomIndex], arr[high]);

	int lastSmaller = low - 1;

	for (int i = 0; i < high; i++)
		if (arr[i] < arr[high])
			std::swap(arr[++lastSmaller], arr[i]);

	std::swap(arr[lastSmaller + 1], arr[high]);

	return lastSmaller + 1;
}


// 内部排序算法定义：快速排序
void quickSort(std::vector<int>& arr, int low, int high)
{
	if (low < high) {
		int pivot = partitionRandomized(arr, low, high);
		quickSort(arr, low, pivot - 1);
		quickSort(arr, pivot + 1, high);
	}
}


// 顺串生成
// inputFile: 测试文件
// CHUNK_SIZE：模拟的内存大小
std::vector<std::string> generateRuns(const std::string intputFile, const int CHUNK_SIZE)
{
	// 记录顺串文件的文件名，用于后续顺串文件的打开与归并排序
	std::vector<std::string> runsFiles;

	std::ifstream infile(intputFile, std::ios::binary);

	int count = 0;

	while (!infile.eof()) {
		// 模拟内存大小
		std::vector<int> chunk(CHUNK_SIZE);

		infile.read(reinterpret_cast<char*>(chunk.data()), CHUNK_SIZE);

		// 确定实际读入的数据量
		std::streamsize elementsRead = infile.gcount() / sizeof(int);
		//std::cout << std::endl << "infile.gcount: " << infile.gcount() << std::endl;
		//std::cout << "elementsRead: " << elementsRead << std::endl;

		// 调整向量长度，便于内部排序
		if (elementsRead == 0) break;
		chunk.resize(elementsRead);

		// 执行内部排序算法，这里使用自主实现的快速排序 --> 最终改用sort，直接使用快排递归深度过大，导致栈溢出
		if (!chunk.empty()) {
			//quickSort(chunk, 0, chunk.size() - 1);
			std::sort(chunk.begin(), chunk.end());
		}


		// 定义顺串文件名
		std::string runsFileName = "run_" + std::to_string(++count) + ".tem";

		std::ofstream outFile(runsFileName, std::ios::binary);
		outFile.write(reinterpret_cast<const char*>(chunk.data()), chunk.size() * sizeof(int));
		outFile.close();

		std::cout << "Runfile ( " << runsFileName << " ) has generated" << std::endl;

		// 记录顺串文件名
		runsFiles.push_back(runsFileName);
	}

	infile.close();

	return runsFiles;
}


// 辅助函数：模拟使用两个输入缓冲区和一个输出缓冲区归并两个文件
void mergeTwoFiles(std::string& inputFile1, std::string& inputFile2, std::string& outputFile)
{
	// 打开文件
	std::ifstream in1(inputFile1, std::ios::binary);
	std::ifstream in2(inputFile2, std::ios::binary);
	std::ofstream out(outputFile, std::ios::binary | std::ios::app);

	// 定义输入/输出缓冲区大小
	const int BUFFER_SIZE = 1024;	// 4KB/sizeof(int)

	// 创建输入/输出缓冲区
	std::vector<int> buffer1(BUFFER_SIZE);
	std::vector<int> buffer2(BUFFER_SIZE);
	std::vector<int> bufferOut(BUFFER_SIZE);

	// 定义索引，指向应该访问的位置
	size_t idx1 = 0, idx2 = 0, idxOut = 0;

	// 填充输入缓冲区
	in1.read(reinterpret_cast<char*>(buffer1.data()), BUFFER_SIZE * sizeof(int));
	size_t size1 = in1.gcount() / sizeof(int);

	in2.read(reinterpret_cast<char*>(buffer2.data()), BUFFER_SIZE * sizeof(int));
	size_t size2 = in2.gcount() / sizeof(int);

	// 只要输入缓冲区非空，就继续归并
	while (idx1 < size1 || idx2 < size2) {
		// 输出缓冲区满，则进行一次IO
		if (idxOut == BUFFER_SIZE) {
			out.write(reinterpret_cast<const char*>(bufferOut.data()), BUFFER_SIZE * sizeof(int));
			idxOut = 0;
		}

		// 若输入缓冲区2耗尽，只能从缓冲区1加载
		// 若缓冲区1元素小于缓冲区2元素，从缓冲区1加载
		if (idx2 >= size2 || (idx1 < size1 && buffer1[idx1] < buffer2[idx2])) {
			bufferOut[idxOut++] = buffer1[idx1++];

			// 更新输入缓冲区
			if (idx1 >= size1) {
				in1.read(reinterpret_cast<char*>(buffer1.data()), BUFFER_SIZE * sizeof(int));
				size1 = in1.gcount() / sizeof(int);
				idx1 = 0;
			}
		}
		else {
			bufferOut[idxOut++] = buffer2[idx2++];

			// 更新输入缓冲区
			if (idx2 >= size2) {
				in2.read(reinterpret_cast<char*>(buffer2.data()), BUFFER_SIZE * sizeof(int));
				size2 = in2.gcount() / sizeof(int);
				idx2 = 0;
			}
		}
	}

	// 将输出缓冲区未写入的剩余数据写入输出文件
	if (idxOut > 0)
		out.write(reinterpret_cast<const char*>(bufferOut.data()), idxOut * sizeof(int));

	// 关闭文件
	in1.close();
	in2.close();
	out.close();
}

// 顺串归并函数
void mergeRuns(std::vector<std::string>& runFiles, const std::string& finalResultFile)
{
	// 记录归并趟数，用于临时顺串文件命名
	int passCount = 0;

	// 直到归并成一个文件
	while (runFiles.size() > 1) {
		std::vector<std::string> nextPassRunFiles;

		for (int i = 0; i < runFiles.size(); i += 2) {
			// 偶数个待归并文件
			if (i + 1 < runFiles.size()) {
				std::string mergedFileName = "pass_" + std::to_string(++passCount) + "_merge_" + std::to_string(i / 2) + ".tem";
				mergeTwoFiles(runFiles[i], runFiles[i + 1], mergedFileName);
				nextPassRunFiles.push_back(mergedFileName);
			}
			// 奇数个待归并文件，直接晋级下一轮归并
			else {
				nextPassRunFiles.push_back(runFiles[i]);
			}
		}

		// 清理临时顺串文件
		for (const std::string& runName : runFiles) {
			std::remove(runName.c_str());
		}

		// 更新顺串文件
		runFiles = nextPassRunFiles;
	}

	// 重命名最终的结果文件
	if (!runFiles.empty())
		std::rename(runFiles[0].c_str(), finalResultFile.c_str());
}


int main()
{
	std::string testFile = "test_file";
	long long recordNum = 100000000;
	const int CHUNK_SIZE = 36 * 1024 * 1024;	// 36MB，刚好可以得到奇数个顺串

	// 测试文件生成
	std::cout << "Start generating testfile... " << std::endl;
	generateTestFile(testFile, recordNum);
	std::cout << "Successfully generate testfile: ";

	/*
	// 快排算法测试
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

	// 顺串生成
	std::cout << "Start generating runsfile... " << std::endl;
	std::vector<std::string> runsFileNames = generateRuns(testFile, CHUNK_SIZE);
	std::cout << "Runs generation process finished" << std::endl << "Successfully generate runsfile: ";
	for (std::string name : runsFileNames)
		std::cout << name << " ";
	std::cout << std::endl;

	// 归并排序
	const std::string resultFileName = "sort_output.bin";
	mergeRuns(runsFileNames, resultFileName);

	// 检查部分归并文件结果
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