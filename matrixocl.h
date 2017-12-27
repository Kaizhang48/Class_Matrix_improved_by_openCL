#ifndef MATRIX
#define MATRIX
#include <iostream>
#include <cmath>
#include <stdlib.h>
#include<vector>
#include "omp.h"
#include <string>
#include <fstream>
#include<CL/cl.h> 
using std::vector;
using std::ostream;
using std::istream;
using std::cout;
using std::cin;
using std::endl;
using std::out_of_range;

cl_platform_id* & get_platforms() {
	cl_int errNum;
	cl_uint numofpt;
	cl_platform_id* platforms = NULL;
	errNum = clGetPlatformIDs(0, NULL, &numofpt);//get the total number of platforms
	if (errNum != CL_SUCCESS || numofpt <= 0) {
		std::cerr << "Failed to find any OpenCL platforms" << std::endl;
	}
	platforms = new cl_platform_id[numofpt];
	errNum = clGetPlatformIDs(numofpt, platforms, NULL);
	if (errNum != CL_SUCCESS || numofpt <= 0) {
		std::cerr << "Failed to find any OpenCL platforms" << std::endl;
	}
	return platforms;
}
cl_device_id*& get_devices(cl_platform_id& pt, const cl_device_type& device_type) {
	cl_int errNum;
	cl_uint numdevices;
	cl_device_id* devices = NULL;
	errNum = clGetDeviceIDs(pt, device_type, 0, NULL, &numdevices);
	if (errNum != CL_SUCCESS || numdevices <= 0) {
		std::cerr << "Failed to find any OpenCL devices" << std::endl;
	}
	devices = new cl_device_id[numdevices];
	errNum = clGetDeviceIDs(pt, device_type, numdevices, devices, NULL);
	if (errNum != CL_SUCCESS || numdevices <= 0) {
		std::cerr << "Failed to find any OpenCL devices" << std::endl;
	}
	return devices;
}

cl_context& creat_context(const cl_platform_id& pt,const cl_device_id& devices, const int& numofdevcies ) {
	cl_context_properties prop[] = { CL_CONTEXT_PLATFORM,(cl_context_properties)pt,0 };
	cl_context context = clCreateContext(prop, numofdevcies, &devices, NULL, NULL, NULL);
	return context;
}

double** init_mat(const int row, const int col, double c = 0) {
	double** d = new double*[row];
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < row; ++i)
			d[i] = new double[col];
	}
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				d[i][j] = c;
			}
		}
	}
	return d;
}

class matrix {
	friend matrix mean(const matrix&);
	friend matrix inv(matrix a);
	friend matrix operator* (const matrix& A, const matrix& B);
	friend matrix operator* (const matrix& A, const double &B);
	friend matrix operator+ (const matrix& A, const matrix &B);
	friend matrix operator- (const matrix& A, const matrix& B);
	friend ostream& operator <<(std::ostream &os, const matrix &m);
	friend cl_platform_id* & get_platforms();
	friend cl_device_id*& get_devices(cl_platform_id&, const cl_device_type&);
	friend cl_context& creat_context(const cl_platform_id& pt, const cl_device_id& devices, const int& numofdevcies);
	friend double** init_mat(const int, const int, double);
	friend double*& t2o(const matrix&);
	friend matrix o2t(const int&, const int&, double*);
public:
	double*& operator[](int t);
	matrix() = default;
	matrix(double* const a, const int&n);
	matrix(const int& a, const int& b, const double& c = 0);
	matrix(const int&, const int&, double*);
	matrix(const matrix &copyfrom);
	matrix(matrix &&movefrom);
	~matrix();
	int getrow()const;
	int getcol()const;
	double getdata(const int&i, const int&j)const;
	matrix& operator=(const matrix& assignfrom);
	matrix& operator=(matrix&& moveassignfrom);
	vector<int> size() const;
	matrix& T();
	double*& t2o();
	void o2t(const int&, const int&, double*);

private:
	int row;
	int col;
	double** data;
	void mfree();

	
};

double*& t2o(const matrix&A) {
	int row = A.row;
	int col = A.col;
	auto sz = row * col;
	double* output = new double[sz];
	auto maxthreads = omp_get_max_threads();
	omp_set_num_threads(maxthreads);
#pragma omp parallel
	{
#pragma omp for
		for (auto i = 0; i < row; ++i) {
			for (auto j = 0; j < col; ++j) {
				output[i*col + j] = A.data[i][j];
			}

		}
	}
	return output;
}
matrix o2t(const int&r, const int&c, double*a) {
	return matrix(r, c, a);
}

double*& matrix::t2o() {
	auto sz = row * col;
	double* output = new double[sz];
	auto maxthreads = omp_get_max_threads();
	omp_set_num_threads(maxthreads);
#pragma omp parallel
	{
#pragma omp for
		for (auto i = 0; i < row; ++i) {
			for (auto j = 0; j < col; ++j) {
				output[i*col + j] = data[i][j];
			}

		}
	}
	return output;
}


void matrix::o2t(const int& r, const int& c, double*a) {
	int Trow = r;
	int Tcol = c;
	double** d = new double*[Trow];
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < Trow; ++i) {
			d[i] = new double[Tcol];
			for (int j = 0; j < Tcol; ++j)
			{
				d[i][j] = a[i*Tcol + j];
			}
		}
	}
	auto temp = data;
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < row; ++i)
		{
			delete[] temp[i];
		}
	}
	delete[] temp;
	row = Trow;
	col = Tcol;
	data = d;
	d = NULL;
}


void matrix::mfree() {
	if (data != nullptr) {
#pragma omp parallel
		{
#pragma omp for
			for (int i = 0; i < row; ++i)
			{
				delete[] data[i];
			}
		}
		delete[] data;
	}
	row = 0;
	col = 0;
}

double*& matrix::operator[](int t) {
	double* &r = data[t];
	return r;
}
matrix::matrix(double* const a, const int&n) :row(1), col(n) {
	data = new double*[1];
	data[0] = new double[n];
	for (int i = 0; i<col; ++i) {
		data[0][i] = a[i];
	}
}
matrix::matrix(const int&a, const int& b, const double& c) :row(a), col(b) {
	data = new double*[row];
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < row; ++i)
			data[i] = new double[col];
	}
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				data[i][j] = c;
			}
		}
	}
}

matrix::matrix(const int&x, const int&y, double*a) :row(x), col(y) {
	data = new double*[row];
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < row; ++i) {
			data[i] = new double[col];
			for (int j = 0; j < col; ++j)
			{
				data[i][j] = a[i*col + j];
			}
		}
	}
}

matrix::matrix(const matrix &copyfrom)
{
	//cout << "this is copy constructor" <<endl;
	row = copyfrom.row;
	col = copyfrom.col;
	data = new double*[row];
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < row; ++i) {
			//printf("I am Thread %d\n", omp_get_thread_num());
			data[i] = new double[col];
		}
	}
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < row; ++i)
		{
			//printf("I am Thread %d\n", omp_get_thread_num());
			for (int j = 0; j < col; ++j)
			{
				data[i][j] = copyfrom.data[i][j];
			}
		}
	}
}
matrix::matrix(matrix &&movefrom) {
	if (movefrom.data != data) {
		data = movefrom.data;
		row = movefrom.row;
		col = movefrom.col;
		movefrom.data = nullptr;
		movefrom.row = 0;
		movefrom.col = 0;
	}
}
matrix& matrix::operator=(const matrix& assignfrom) {
	if (assignfrom.data != data) {
		if (data != nullptr) {
			mfree();
		}
		row = assignfrom.row;
		col = assignfrom.col;
		data = new double*[row];
#pragma omp parallel
		{
#pragma omp for
			for (int i = 0; i < row; ++i)
				data[i] = new double[col];
		}
#pragma omp parallel
		{
#pragma omp for
			for (int i = 0; i < row; ++i)
			{
				for (int j = 0; j < col; ++j)
				{
					data[i][j] = assignfrom.data[i][j];
				}
			}
		}
	}
	return *this;
}

matrix& matrix::operator=(matrix&& moveassignfrom) {
	if (moveassignfrom.data != data) {
		if (data != nullptr) {
			mfree();
		}
		data = moveassignfrom.data;
		row = moveassignfrom.row;
		col = moveassignfrom.col;
		moveassignfrom.data = nullptr;
		moveassignfrom.row = 0;
		moveassignfrom.col = 0;
	}
	return *this;
}

matrix::~matrix()
{
	mfree();
}

int matrix::getrow()const {
	return row;
}

int matrix::getcol()const {
	return col;
}

double matrix::getdata(const int&i, const int&j)const {
	return data[i][j];
}

vector<int> matrix::size() const {
	return vector<int>{row, col};
}

matrix& matrix::T() {
	
	const char *programSouce =
		"__kernel                                         \n"
		"void corefunc(const int r,                       \n"
		"            const int c,                         \n"
		"            __global double *d,                  \n"
		"            __global double *Td)                 \n"
		"{                                                \n"
		"  int x = get_global_id(0);                      \n"
		"  int y = get_global_id(1);                      \n"
		"  if ((x<r) && (y<c))                            \n"
		"   {                                             \n"
		"      Td[y*r+x] = d[x*c+y];                      \n"
		"   }                                             \n"
		"}                                                \n"
		;
	cl_int err;
	cl_platform_id* pts = get_platforms();
	cl_platform_id platform = pts[0];
	cl_device_id* devices = get_devices(platform, CL_DEVICE_TYPE_GPU);
	cl_device_id device = devices[0];
	cl_context context = creat_context(platform, device, 1);
	cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
	auto sz = col * row;

	double* d = t2o();

	double*Td = new double[sz];
	int r = row;
	int c = col;
	cl_mem a_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*sz, NULL, NULL);
	cl_mem c_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double)*sz, NULL, NULL);

	err = clEnqueueWriteBuffer(queue, a_in, CL_TRUE, 0, sizeof(double)*sz, d, 0, NULL, NULL);

	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&programSouce, NULL, &err);

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	if (err != CL_SUCCESS) {
		char buildlog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildlog), buildlog, NULL);
		std::cerr << "error in kernel: " << std::endl;
		std::cerr << buildlog;
		clReleaseProgram(program);
		throw out_of_range("error in kernel");
	}

	cl_kernel kernel = clCreateKernel(program, "corefunc", &err);

	err = 0;
	err = clSetKernelArg(kernel, 0, sizeof(int), &r);
	err = clSetKernelArg(kernel, 1, sizeof(int), &c);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &a_in);
	err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_out);

	size_t* global = new size_t[2];
	global[0] = (size_t)r;
	global[1] = (size_t)c;
	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
	clFinish(queue);
	err = clEnqueueReadBuffer(queue, c_out, CL_TRUE, 0, sizeof(double)*sz, Td, 0, NULL, NULL);

	o2t(col, row, Td);
	delete[] global;
	delete[]d;
	delete[]Td;
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(a_in);
	clReleaseMemObject(c_out);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	return *this;
}



matrix inv(matrix b)
{
	double**&a = b.data;
	int& n = b.row;
	int *is = new int[n];
	int *js = new int[n];
	int i, j, k;
	double d, p;
	for (k = 0; k < n; k++)
	{
		d = 0.0;
		for (i = k; i <= n - 1; i++)
			for (j = k; j <= n - 1; j++)
			{
				p = fabs(a[i][j]);
				if (p>d) { d = p; is[k] = i; js[k] = j; }
			}
		if (0.0 == d)
		{
			free(is); free(js);
			throw out_of_range("can not be inversed !");
		}
		if (is[k] != k)
			for (j = 0; j <= n - 1; j++)
			{
				p = a[k][j];
				a[k][j] = a[is[k]][j];
				a[is[k]][j] = p;
			}
		if (js[k] != k)
			for (i = 0; i <= n - 1; i++)
			{
				p = a[i][k];
				a[i][k] = a[i][js[k]];
				a[i][js[k]] = p;
			}
		a[k][k] = 1.0 / a[k][k];
		for (j = 0; j <= n - 1; j++)
			if (j != k)
			{
				a[k][j] *= a[k][k];
			}
		for (i = 0; i <= n - 1; i++)
			if (i != k)
				for (j = 0; j <= n - 1; j++)
					if (j != k)
					{
						a[i][j] -= a[i][k] * a[k][j];
					}
		for (i = 0; i <= n - 1; i++)
			if (i != k)
			{
				a[i][k] = -a[i][k] * a[k][k];
			}
	}


	for (k = n - 1; k >= 0; k--)
	{
		if (js[k] != k)
			for (j = 0; j <= n - 1; j++)
			{
				p = a[k][j];
				a[k][j] = a[js[k]][j];
				a[js[k]][j] = p;
			}
		if (is[k] != k)
			for (i = 0; i <= n - 1; i++)
			{
				p = a[i][k];
				a[i][k] = a[i][is[k]];
				a[i][is[k]] = p;
			}
	}

	free(is); free(js);
	return std::move(b);
}

matrix operator*(const matrix& A, const matrix& B) {
	const char *programSouce =
		"__kernel                                         \n"
		"void corefunc(const int M,                       \n"
		"            const int N,                         \n"
		"            const int P,                         \n"
		"            __global double *a,                  \n"
		"            __global double *b,                  \n"
		"            __global double *c)                  \n"
		"{                                                \n"
		"  int k,j;                                       \n"
		"  int i = get_global_id(0);                      \n"
		"  double tmp;                                    \n"
		"  if (i<N)                                       \n"
		"   {                                             \n"
		"		for (j=0;j<M;++j){                        \n"
		"			tmp=0;                                \n"
		"			for (k=0;k<P;++k)                     \n"
		"				tmp+=(a[i*P+k]*b[k*M+j]);         \n"
		"			c[i*M+j]=tmp;                         \n"
		"		}                                         \n"
		"   }                                             \n"
		"}                                                \n"
		;
	int N = A.getrow();
	int P = A.getcol();

	int PB = B.getrow();
	int M = B.getcol();
	if (P!=PB) {
		throw out_of_range("two matrices do not match the size!");
	}
	auto szA = N * P;
	auto szB = P * M;
	auto szC = N * M;

	cl_int err;
	cl_platform_id* pts = get_platforms();
	cl_platform_id platform = pts[0];
	cl_device_id* devices = get_devices(platform, CL_DEVICE_TYPE_GPU);
	cl_device_id device = devices[0];
	cl_context context = creat_context(platform, device, 1);
	cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

	double*a = t2o(A);
	double*b = t2o(B);

	double*c = new double[szC];

	cl_mem a_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*szA, NULL, NULL);
	cl_mem b_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*szB, NULL, NULL);
	cl_mem c_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double)*szC, NULL, NULL);

	err = clEnqueueWriteBuffer(queue, a_in, CL_TRUE, 0, sizeof(double)*szA, a, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, b_in, CL_TRUE, 0, sizeof(double)*szB, b, 0, NULL, NULL);

	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&programSouce, NULL, &err);

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	if (err != CL_SUCCESS) {
		char buildlog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildlog), buildlog, NULL);
		std::cerr << "error in kernel: " << std::endl;
		std::cerr << buildlog;
		clReleaseProgram(program);
		throw out_of_range("error in kernel");
	}

	cl_kernel kernel = clCreateKernel(program, "corefunc", &err);

	err = 0;
	err = clSetKernelArg(kernel, 0, sizeof(int), &M);
	err = clSetKernelArg(kernel, 1, sizeof(int), &N);
	err = clSetKernelArg(kernel, 2, sizeof(int), &P);
	err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &a_in);
	err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &b_in);
	err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &c_out);

	size_t* global = new size_t[1];
	global[0] = (size_t)N;

	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, NULL);
	clFinish(queue);
	err = clEnqueueReadBuffer(queue, c_out, CL_TRUE, 0, sizeof(double)*szC, c, 0, NULL, NULL);

	matrix C = o2t(N, M, c);
	delete[] global;
	delete[]a;
	delete[]b;
	delete[]c;
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(a_in);
	clReleaseMemObject(b_in);
	clReleaseMemObject(c_out);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return std::move(C);
}

matrix operator*(matrix& A, const double &B) {

	const char *programSouce =
		"__kernel                                         \n"
		"void corefunc(const int sz,                      \n"
		"             const double b,                     \n"
		"            __global double *a,                  \n"
		"            __global double *c)                  \n"
		"{                                                \n"
		"  int idx = get_global_id(0);                    \n"
		"  if (idx<sz)                                    \n"
		"   {                                             \n"
		"      c[idx]=a[idx]*b;                           \n"
		"   }                                             \n"
		"}                                                \n"
		;
	int row = A.getrow();
	int col = A.getcol();
	auto sz = col * row;

	cl_int err;
	cl_platform_id* pts = get_platforms();
	cl_platform_id platform = pts[0];
	cl_device_id* devices = get_devices(platform, CL_DEVICE_TYPE_GPU);
	cl_device_id device = devices[0];
	cl_context context = creat_context(platform, device, 1);
	cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

	double*a = t2o(A);
	double b =B;
	double*c = new double[sz];

	cl_mem a_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*sz, NULL, NULL);
	cl_mem c_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double)*sz, NULL, NULL);

	err = clEnqueueWriteBuffer(queue, a_in, CL_TRUE, 0, sizeof(double)*sz, a, 0, NULL, NULL);

	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&programSouce, NULL, &err);

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	if (err != CL_SUCCESS) {
		char buildlog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildlog), buildlog, NULL);
		std::cerr << "error in kernel: " << std::endl;
		std::cerr << buildlog;
		clReleaseProgram(program);
		throw out_of_range("error in kernel");
	}

	cl_kernel kernel = clCreateKernel(program, "corefunc", &err);

	err = 0;
	err = clSetKernelArg(kernel, 0, sizeof(int), &sz);
	err = clSetKernelArg(kernel, 1, sizeof(double), &b);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &a_in);
	err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_out);

	size_t* global = new size_t[1];
	global[0] = (size_t)sz;
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, NULL);
	clFinish(queue);
	err = clEnqueueReadBuffer(queue, c_out, CL_TRUE, 0, sizeof(double)*sz, c, 0, NULL, NULL);

	matrix C = o2t(row, col, c);
	delete[] global;
	delete[]a;
	delete[]c;
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(a_in);
	clReleaseMemObject(c_out);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return std::move(C);
}

matrix operator*(const double &B, matrix& A) {
	return std::move(A*B);
}

matrix operator+ (const matrix& A, const matrix &B) {

	const char *programSouce =
		"__kernel                                         \n"
		"void corefunc(const int sz,                      \n"
		"            __global double *a,                  \n"
		"            __global double *b,                  \n"
		"            __global double *c)                  \n"
		"{                                                \n"
		"  int idx = get_global_id(0);                    \n"
		"  if (idx<sz)                                    \n"
		"   {                                             \n"
		"      c[idx]=a[idx]+b[idx];                      \n"
		"   }                                             \n"
		"}                                                \n"
		;
	int row = A.getrow();
	int col = A.getcol();

	int Brow = B.getrow();
	int Bcol = B.getcol();
	if (row != Brow || col != Bcol) {
		throw out_of_range("two matrices should have the same size!");
	}
	auto sz = col * row;

	cl_int err;
	cl_platform_id* pts = get_platforms();
	cl_platform_id platform = pts[0];
	cl_device_id* devices = get_devices(platform, CL_DEVICE_TYPE_GPU);
	cl_device_id device = devices[0];
	cl_context context = creat_context(platform, device, 1);
	cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
	
	double*a = t2o(A);
	double*b = t2o(B);
	double*c = new double[sz];

	cl_mem a_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*sz, NULL, NULL);
	cl_mem b_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*sz, NULL, NULL);
	cl_mem c_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double)*sz, NULL, NULL);

	err = clEnqueueWriteBuffer(queue, a_in, CL_TRUE, 0, sizeof(double)*sz, a, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, b_in, CL_TRUE, 0, sizeof(double)*sz, b, 0, NULL, NULL);

	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&programSouce, NULL, &err);

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	if (err != CL_SUCCESS) {
		char buildlog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildlog), buildlog, NULL);
		std::cerr << "error in kernel: " << std::endl;
		std::cerr << buildlog;
		clReleaseProgram(program);
		throw out_of_range("error in kernel");
	}

	cl_kernel kernel = clCreateKernel(program, "corefunc", &err);

	err = 0;
	err = clSetKernelArg(kernel, 0, sizeof(int), &sz);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &a_in);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &b_in);
	err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_out);

	size_t* global = new size_t[1];
	global[0] = (size_t)sz;
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, NULL);
	clFinish(queue);
	err = clEnqueueReadBuffer(queue, c_out, CL_TRUE, 0, sizeof(double)*sz, c, 0, NULL, NULL);

	matrix C=o2t(row, col, c);
	delete[] global;
	delete[]a;
	delete[]b;
	delete[]c;
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(a_in);
	clReleaseMemObject(b_in);
	clReleaseMemObject(c_out);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return std::move(C);
}


matrix operator- (matrix& A, matrix& B) {
	const char *programSouce =
		"__kernel                                         \n"
		"void corefunc(const int sz,                      \n"
		"            __global double *a,                  \n"
		"            __global double *b,                  \n"
		"            __global double *c)                  \n"
		"{                                                \n"
		"  int idx = get_global_id(0);                    \n"
		"  if (idx<sz)                                    \n"
		"   {                                             \n"
		"      c[idx]=a[idx]-b[idx];                      \n"
		"   }                                             \n"
		"}                                                \n"
		;
	int row = A.getrow();
	int col = A.getcol();

	int Brow = B.getrow();
	int Bcol = B.getcol();
	if (row != Brow || col != Bcol) {
		throw out_of_range("two matrices should have the same size!");
	}
	auto sz = col * row;

	cl_int err;
	cl_platform_id* pts = get_platforms();
	cl_platform_id platform = pts[0];
	cl_device_id* devices = get_devices(platform, CL_DEVICE_TYPE_GPU);
	cl_device_id device = devices[0];
	cl_context context = creat_context(platform, device, 1);
	cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

	double*a = t2o(A);
	double*b = t2o(B);
	double*c = new double[sz];

	cl_mem a_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*sz, NULL, NULL);
	cl_mem b_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*sz, NULL, NULL);
	cl_mem c_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double)*sz, NULL, NULL);

	err = clEnqueueWriteBuffer(queue, a_in, CL_TRUE, 0, sizeof(double)*sz, a, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, b_in, CL_TRUE, 0, sizeof(double)*sz, b, 0, NULL, NULL);

	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&programSouce, NULL, &err);

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	if (err != CL_SUCCESS) {
		char buildlog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildlog), buildlog, NULL);
		std::cerr << "error in kernel: " << std::endl;
		std::cerr << buildlog;
		clReleaseProgram(program);
		throw out_of_range("error in kernel");
	}

	cl_kernel kernel = clCreateKernel(program, "corefunc", &err);

	err = 0;
	err = clSetKernelArg(kernel, 0, sizeof(int), &sz);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &a_in);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &b_in);
	err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_out);

	size_t* global = new size_t[1];
	global[0] = (size_t)sz;
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, NULL);
	clFinish(queue);
	err = clEnqueueReadBuffer(queue, c_out, CL_TRUE, 0, sizeof(double)*sz, c, 0, NULL, NULL);

	matrix C = o2t(row, col, c);
	delete[] global;
	delete[]a;
	delete[]b;
	delete[]c;
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(a_in);
	clReleaseMemObject(b_in);
	clReleaseMemObject(c_out);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return std::move(C);
}

ostream& operator <<(std::ostream &os, const matrix &m)
{
	for (int i = 0; i < m.row; i++)
	{
		os << " | ";
		for (int j = 0; j < m.col; j++)
		{
			char buf[32];
			double data = m.data[i][j];
			if (m.data[i][j] > -0.00001 && m.data[i][j] < 0.00001)
				data = 0;
			sprintf_s(buf, "%10.10lf ", data);
			os << buf;

		}
		os << "|\n";
	}
	os << "\n\n";
	return os;
}


matrix mean(const matrix& A) {
	int row= A.getrow();
	int col = A.getcol();
	const char *programSouce =
		"__kernel                                         \n"
		"void corefunc(const int row,                     \n"
		"            const int col,                       \n"
		"            __global double *a,                  \n"
		"            __global double *c)                  \n"
		"{                                                \n"
		"  int j;                                         \n"
		"  int i = get_global_id(0);                      \n"
		"  double tmp;                                    \n"
		"  if (i<row)                                     \n"
		"   {                                             \n"
		"		tmp=0;                                    \n"
		"		for (j=0;j<col;++j)                       \n"
		"			tmp+=a[i*col+j];                      \n"
		"		c[i]=tmp/col;                             \n"
		"   }                                             \n"
		"}                                                \n"
		;

	auto sz = row*col;

	cl_int err;
	cl_platform_id* pts = get_platforms();
	cl_platform_id platform = pts[0];
	cl_device_id* devices = get_devices(platform, CL_DEVICE_TYPE_GPU);
	cl_device_id device = devices[0];
	cl_context context = creat_context(platform, device, 1);
	cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

	double*a = t2o(A);

	double*c = new double[row];

	cl_mem a_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*sz, NULL, NULL);

	cl_mem c_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double)*row, NULL, NULL);

	err = clEnqueueWriteBuffer(queue, a_in, CL_TRUE, 0, sizeof(double)*sz, a, 0, NULL, NULL);

	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&programSouce, NULL, &err);

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	if (err != CL_SUCCESS) {
		char buildlog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildlog), buildlog, NULL);
		std::cerr << "error in kernel: " << std::endl;
		std::cerr << buildlog;
		clReleaseProgram(program);
		throw out_of_range("error in kernel");
	}

	cl_kernel kernel = clCreateKernel(program, "corefunc", &err);

	err = 0;
	err = clSetKernelArg(kernel, 0, sizeof(int), &row);
	err = clSetKernelArg(kernel, 1, sizeof(int), &col);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &a_in);
	err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_out);

	size_t* global = new size_t[1];
	global[0] = (size_t)row;

	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, NULL);
	clFinish(queue);
	err = clEnqueueReadBuffer(queue, c_out, CL_TRUE, 0, sizeof(double)*row, c, 0, NULL, NULL);
	matrix C = o2t(row, 1, c);
	delete[] global;
	delete[]a;
	delete[]c;
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(a_in);
	clReleaseMemObject(c_out);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return std::move(C);
}

void removemean(matrix& A) {
	matrix miu = mean(A);
	int row = A.getrow();
	int col = A.getcol();
	const char *programSouce =
		"__kernel                                         \n"
		"void corefunc(const int row,                     \n"
		"            const int col,                       \n"
		"            __global double *a,                  \n"
		"            __global double *c)                  \n"
		"{                                                \n"
		"  int j;                                         \n"
		"  int i = get_global_id(0);                      \n"
		"  if (i<row)                                     \n"
		"   {                                             \n"
		"		for (j=0;j<col;++j)                       \n"
		"			c[i*col+j]=a[i];                      \n"
		"   }                                             \n"
		"}                                                \n"
		;

	auto sz = row * col;

	cl_int err;
	cl_platform_id* pts = get_platforms();
	cl_platform_id platform = pts[0];
	cl_device_id* devices = get_devices(platform, CL_DEVICE_TYPE_GPU);
	cl_device_id device = devices[0];
	cl_context context = creat_context(platform, device, 1);
	cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

	double*a = t2o(miu);

	double*c = new double[sz];

	cl_mem a_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*row, NULL, NULL);

	cl_mem c_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double)*sz, NULL, NULL);

	err = clEnqueueWriteBuffer(queue, a_in, CL_TRUE, 0, sizeof(double)*row, a, 0, NULL, NULL);

	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&programSouce, NULL, &err);

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	if (err != CL_SUCCESS) {
		char buildlog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildlog), buildlog, NULL);
		std::cerr << "error in kernel: " << std::endl;
		std::cerr << buildlog;
		clReleaseProgram(program);
		throw out_of_range("error in kernel");
	}

	cl_kernel kernel = clCreateKernel(program, "corefunc", &err);

	err = 0;
	err = clSetKernelArg(kernel, 0, sizeof(int), &row);
	err = clSetKernelArg(kernel, 1, sizeof(int), &col);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &a_in);
	err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_out);

	size_t* global = new size_t[1];
	global[0] = (size_t)row;

	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, NULL);
	clFinish(queue);
	err = clEnqueueReadBuffer(queue, c_out, CL_TRUE, 0, sizeof(double)*sz, c, 0, NULL, NULL);
	matrix C=o2t(row, col, c);
	A = A - C;
	delete[] global;
	delete[]a;
	delete[]c;
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(a_in);
	clReleaseMemObject(c_out);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}

matrix cov(matrix a) {
	removemean(a);
	matrix aT = a;
	aT.T();
	matrix result = a * aT;
	result = result * (1.0 / (a.getcol() - 1));
	return std::move(result);
}


#endif