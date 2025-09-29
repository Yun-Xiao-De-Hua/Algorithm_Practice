#include<iostream>
#include<fstream>
#include<string>
#include<random>
#include<chrono>
#include<vector>
#include<utility>


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


int main()
{
	std::string testFile = "test_file";
	long long recordNum = 100000000;
	const int CHUNK_SIZE = 36 * 1024 * 1024;	// 36MB，刚好可以得到奇数个顺串

	// 测试文件生成
	std::cout << "Start generating testfile... " << std::endl;
	generateTestFile(testFile, recordNum);
	std::cout << "Successfully generate testfile: ";

	//// 快排算法测试
	//std::vector<int> testArr = { 2,3,1,4,6,5,9,7,0,8 };

	//std::cout << "origin: ";
	//for (int ele : testArr)
	//	std::cout << ele << " ";

	//quickSort(testArr, 0, testArr.size() - 1);

	//std::cout << std::endl;
	//std::cout << "result: ";
	//for (int ele : testArr)
	//	std::cout << ele << " ";

	// 顺串生成
	std::cout << "Start generating runsfile... " << std::endl;
	std::vector<std::string> runsFileNames = generateRuns(testFile, CHUNK_SIZE);
	std::cout << "Runs generation process finished" << std::endl << "Successfully generate runsfile: ";
	for (std::string name : runsFileNames)
		std::cout << name << " ";
	std::cout << std::endl;

	// 归并排序




	return 0;
}