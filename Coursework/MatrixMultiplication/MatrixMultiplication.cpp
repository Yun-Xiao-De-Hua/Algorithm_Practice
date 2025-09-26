#include<iostream>
#include<vector>
#include<random>
#include<chrono>
#include<iomanip>
#include<cstring>

using matrix = std::vector<std::vector<int>>;


// --------------------�������ߺ���--------------------

// �������
void printMatrix(const matrix& input)
{
	std::cout << "Matrix: " << std::endl;

	for (int i = 0; i < input.size(); i++){
		for (int j = 0; j < input[0].size(); j++)
			std::cout << input[i][j] << " ";
		std::cout << std::endl;
	}
}


// �����ʼ���������ֵ
void initializeMatrix(matrix& input, std::mt19937& gen)
{
	// ����һ���ֲ��������ȷֲ�
	std::uniform_int_distribution<> distrib(0, 500);

	// �����ֵ
	for (int i = 0; i < input.size(); i++)
		for (int j = 0; j < input[0].size(); j++)
			input[i][j] = distrib(gen);
}



// --------------------����6�ֲ�ͬ���㷽ʽ--------------------

// ijk Order
matrix calculateByIJK(matrix&A, matrix&B,int m, int p, int n)
{
	// ����������
	matrix result(m,std::vector<int>(n,0));

	// �������˳��
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			for (int k = 0; k < p; k++)
				result[i][j] += A[i][k] * B[k][j];

	// ���ؽ������
	return result;
}


// ikj Order
matrix calculateByIKJ(matrix& A, matrix& B, int m, int p, int n)
{
	// ����������
	matrix result(m, std::vector<int>(n, 0));

	// �������˳��
	for (int i = 0; i < m; i++)
		for (int k = 0; k < p; k++)
			for (int j = 0; j < n; j++)
				result[i][j] += A[i][k] * B[k][j];

	// ���ؽ������
	return result;
}


// jik Order
matrix calculateByJIK(matrix& A, matrix& B, int m, int p, int n)
{
	// ����������
	matrix result(m, std::vector<int>(n, 0));

	// �������˳��
	for (int j = 0; j < n; j++)
		for (int i = 0; i < m; i++)
			for (int k = 0; k < p; k++)
				result[i][j] += A[i][k] * B[k][j];

	// ���ؽ������
	return result;
}


// jki Order
matrix calculateByJKI(matrix& A, matrix& B, int m, int p, int n)
{
	// ����������
	matrix result(m, std::vector<int>(n, 0));

	// �������˳��
	for (int j = 0; j < n; j++)
		for (int k = 0; k < p; k++)
			for (int i = 0; i < m; i++)
				result[i][j] += A[i][k] * B[k][j];

	// ���ؽ������
	return result;
}


// kij Order
matrix calculateByKIJ(matrix& A, matrix& B, int m, int p, int n)
{
	// ����������
	matrix result(m, std::vector<int>(n, 0));

	// �������˳��
	for (int k = 0; k < p; k++)
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				result[i][j] += A[i][k] * B[k][j];

	// ���ؽ������
	return result;
}


// kji Order
matrix calculateByKJI(matrix& A, matrix& B, int m, int p, int n)
{
	// ����������
	matrix result(m, std::vector<int>(n, 0));

	// �������˳��
	for (int k = 0; k < p; k++)
		for (int j = 0; j < n; j++)
			for (int i = 0; i < m; i++)
				result[i][j] += A[i][k] * B[k][j];

	// ���ؽ������
	return result;
}


int main()
{
	// ������Ծ���A��B����Ӧ�������ΪC
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

	// �������Ծ���
	matrix A(m,std::vector<int>(p,0));
	matrix B(p, std::vector<int>(n, 0));

	// ��ʼ�����Ծ���
	std::random_device rd;
	std::mt19937 gen(rd());

	initializeMatrix(A, gen);
	initializeMatrix(B, gen);

	// ������ʱ
	auto measureTime = [&](const std::string& method, auto func)
		{
			std::cout << "Running " << method << " order..." << std::endl;

			auto start = std::chrono::high_resolution_clock::now();
			matrix C = func(A, B, m, p, n);
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> diff = end - start;

			std::cout << "Time taken: " << std::fixed << std::setprecision(4) << diff.count() << std::endl;
		};

	// ����6�ֲ�ͬ���㷽ʽ��ʱ
	measureTime("ijk", calculateByIJK);
	measureTime("ikj", calculateByIKJ);
	measureTime("kij", calculateByKIJ);
	measureTime("kji", calculateByKJI);
	measureTime("jik", calculateByJIK);
	measureTime("jki", calculateByJKI);

	return 0;
}