#include<iostream>
#include<vector>
#include<random>
#include<chrono>
#include<iomanip>
#include<cstring>

using matrix = std::vector<std::vector<int>>;


// --------------------辅助工具函数--------------------

// 矩阵输出
void printMatrix(const matrix& input)
{
	std::cout << "Matrix: " << std::endl;

	for (int i = 0; i < input.size(); i++){
		for (int j = 0; j < input[0].size(); j++)
			std::cout << input[i][j] << " ";
		std::cout << std::endl;
	}
}


// 矩阵初始化，随机赋值
void initializeMatrix(matrix& input, std::mt19937& gen)
{
	// 创建一个分布器，均匀分布
	std::uniform_int_distribution<> distrib(0, 500);

	// 随机赋值
	for (int i = 0; i < input.size(); i++)
		for (int j = 0; j < input[0].size(); j++)
			input[i][j] = distrib(gen);
}



// --------------------定义6种不同计算方式--------------------

// ijk Order
matrix calculateByIJK(matrix&A, matrix&B,int m, int p, int n)
{
	// 定义结果矩阵
	matrix result(m,std::vector<int>(n,0));

	// 定义遍历顺序
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			for (int k = 0; k < p; k++)
				result[i][j] += A[i][k] * B[k][j];

	// 返回结果矩阵
	return result;
}


// ikj Order
matrix calculateByIKJ(matrix& A, matrix& B, int m, int p, int n)
{
	// 定义结果矩阵
	matrix result(m, std::vector<int>(n, 0));

	// 定义遍历顺序
	for (int i = 0; i < m; i++)
		for (int k = 0; k < p; k++)
			for (int j = 0; j < n; j++)
				result[i][j] += A[i][k] * B[k][j];

	// 返回结果矩阵
	return result;
}


// jik Order
matrix calculateByJIK(matrix& A, matrix& B, int m, int p, int n)
{
	// 定义结果矩阵
	matrix result(m, std::vector<int>(n, 0));

	// 定义遍历顺序
	for (int j = 0; j < n; j++)
		for (int i = 0; i < m; i++)
			for (int k = 0; k < p; k++)
				result[i][j] += A[i][k] * B[k][j];

	// 返回结果矩阵
	return result;
}


// jki Order
matrix calculateByJKI(matrix& A, matrix& B, int m, int p, int n)
{
	// 定义结果矩阵
	matrix result(m, std::vector<int>(n, 0));

	// 定义遍历顺序
	for (int j = 0; j < n; j++)
		for (int k = 0; k < p; k++)
			for (int i = 0; i < m; i++)
				result[i][j] += A[i][k] * B[k][j];

	// 返回结果矩阵
	return result;
}


// kij Order
matrix calculateByKIJ(matrix& A, matrix& B, int m, int p, int n)
{
	// 定义结果矩阵
	matrix result(m, std::vector<int>(n, 0));

	// 定义遍历顺序
	for (int k = 0; k < p; k++)
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				result[i][j] += A[i][k] * B[k][j];

	// 返回结果矩阵
	return result;
}


// kji Order
matrix calculateByKJI(matrix& A, matrix& B, int m, int p, int n)
{
	// 定义结果矩阵
	matrix result(m, std::vector<int>(n, 0));

	// 定义遍历顺序
	for (int k = 0; k < p; k++)
		for (int j = 0; j < n; j++)
			for (int i = 0; i < m; i++)
				result[i][j] += A[i][k] * B[k][j];

	// 返回结果矩阵
	return result;
}


// --------------------统计缓存缺失次数--------------------

long long calCacheMissByIJK(int m, int p, int n, int w)
{
	long long miss = 0;

	// A缺失次数
	miss += (long long)(m * n * p / w);

	// B缺失次数
	miss += (long long)(m * n * p);

	// C缺失次数
	miss += (long long)(m * n / w);

	return miss;
}

long long calCacheMissByIKJ(int m, int p, int n, int w)
{
	long long miss = 0;

	// A缺失次数
	miss += (long long)(m * p / w);

	// B缺失次数
	miss += (long long)(m * n * p / w);

	// C缺失次数
	miss += (long long)(m * n / w);

	return miss;
}

long long calCacheMissByJIK(int m, int p, int n, int w)
{
	long long miss = 0;

	// A缺失次数
	miss += (long long)(n * m * p / w);

	// B缺失次数
	miss += (long long)(n * m * p);

	// C缺失次数
	miss += (long long)(n * m);

	return miss;
}

long long calCacheMissByJKI(int m, int p, int n, int w)
{
	long long miss = 0;

	// A缺失次数
	miss += (long long)(n * p * m);

	// B缺失次数
	miss += (long long)(n * p);

	// C缺失次数
	miss += (long long)(n * p * m);

	return miss;
}

long long calCacheMissByKIJ(int m, int p, int n, int w)
{
	long long miss = 0;

	// A缺失次数
	miss += (long long)(p * m);

	// B缺失次数
	miss += (long long)(p * m * n / w);

	// C缺失次数
	miss += (long long)(p * m * n / w);

	return miss;
}

long long calCacheMissByKJI(int m, int p, int n, int w)
{
	long long miss = 0;

	// A缺失次数
	miss += (long long)(p * n * m);

	// B缺失次数
	miss += (long long)(p * n / w);

	// C缺失次数
	miss += (long long)(p * n * m);

	return miss;
}


int main()
{
	// 定义测试矩阵A，B，对应结果矩阵为C
	// A: m*p
	// B: p*n
	// C: m*n

	const int m = 512;
	const int p = 256;
	const int n = 1014;

	std::cout << "Matrix Dimensions:" << std::endl;
	std::cout << "  A: " << m << " x " << p << std::endl;
	std::cout << "  B: " << p << " x " << n << std::endl;
	std::cout << "  C: " << m << " x " << n << std::endl << std::endl;


	// 创建测试矩阵
	matrix A(m,std::vector<int>(p,0));
	matrix B(p, std::vector<int>(n, 0));

	// 初始化测试矩阵
	std::random_device rd;
	std::mt19937 gen(rd());

	initializeMatrix(A, gen);
	initializeMatrix(B, gen);

	// 计算用时
	auto measureTime = [&](const std::string& method, auto func)
		{
			std::cout << "Running " << method << " order..." << std::endl;

			auto start = std::chrono::high_resolution_clock::now();
			matrix C = func(A, B, m, p, n);
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> diff = end - start;

			std::cout << "Time taken: " << std::fixed << std::setprecision(4) << diff.count() << std::endl;
		};

	// 测试6种不同计算方式用时
	measureTime("ijk", calculateByIJK);
	measureTime("ikj", calculateByIKJ);
	measureTime("kij", calculateByKIJ);
	measureTime("kji", calculateByKJI);
	measureTime("jik", calculateByJIK);
	measureTime("jki", calculateByJKI);

	// 统计6种不同计算方式缓存缺失次数
	int w = 1;
	auto measureMiss = [&](const std::string& method, auto func)
		{
			long long miss = 0;
			miss = func(m, p, n, w);
			std::cout << "(Case" << w << ") Cache line width: " << w << " elements" << std::endl;
			std::cout << "Running " << method << " order..." << std::endl;
			std::cout << "Cache miss times: " << miss << std::endl;
		};

	std::cout << std::endl << "Start cache miss times test for w ranging from 1 to 10: " << std::endl;

	for (w; w <= 10; w++) {
		measureMiss("ijk", calCacheMissByIJK);
		measureMiss("ikj", calCacheMissByIKJ);
		measureMiss("JIK", calCacheMissByJIK);
		measureMiss("JKI", calCacheMissByJKI);
		measureMiss("KIJ", calCacheMissByKIJ);
		measureMiss("KJI", calCacheMissByKJI);
	}

	return 0;
}