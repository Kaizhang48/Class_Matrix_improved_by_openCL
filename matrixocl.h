#pragma once
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

cl_platform_id* & get_platforms();
cl_device_id*& get_devices(cl_platform_id&, const cl_device_type& );
cl_context& creat_context(const cl_platform_id&, const cl_device_id&, const int&);

class matrix {
	friend matrix mean(const matrix&);
	friend matrix inv(matrix);
	friend matrix operator*(matrix&, const double &);
	friend matrix operator* (const matrix& , const matrix& );
	friend matrix operator+ (const matrix& , const matrix&);
	friend matrix operator- (const matrix& , const matrix& );
	friend matrix operator- (matrix& , matrix& );
	friend ostream& operator <<(std::ostream &, const matrix &);
	friend cl_platform_id* & get_platforms();
	friend cl_device_id*& get_devices(cl_platform_id&, const cl_device_type&);
	friend cl_context& creat_context(const cl_platform_id& , const cl_device_id&, const int&);
	friend void removemean(matrix&);
public:
	//double*& operator[](int t);
	matrix() = default;
	matrix(double* const a, const int&n);
	matrix(const int& a, const int& b, const double& c = 0);
	matrix(const int&, const int&, double*);
	matrix(const matrix &copyfrom);
	matrix(matrix &&movefrom);
	~matrix();
	int getrow()const;
	int getcol()const;
	double& getdata(const int&i, const int&j);
	matrix& operator=(const matrix& assignfrom);
	matrix& operator=(matrix&& moveassignfrom);
	vector<int> size() const;
	matrix& T();
	double*& t2o();
	void o2t(const int&, const int&, double*);

private:
	int row;
	int col;
	double* data;
	void mfree();


};

void matrix::mfree() {
	if (data != nullptr) {
		delete[] data;
	}
	data = nullptr;
	row = 0;
	col = 0;
}

matrix::matrix(const int&a, const int& b, const double& c) :row(a), col(b) {
	data = new double[row*col];
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				data[i*col+j] = c;
			}
		}
	}
}

matrix::matrix(const int&x, const int&y, double*a) :row(x), col(y) {
	data = new double[row*col];
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < row; ++i) {
			for (int j = 0; j < col; ++j)
			{
				data[i*col+j] = a[i*col + j];
			}
		}
	}
}

matrix::matrix(const matrix &copyfrom)
{
	row = copyfrom.row;
	col = copyfrom.col;
	data = new double[row*col];

#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				data[i*col+j] = copyfrom.data[i*col+j];
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
		data = new double[row*col];

#pragma omp parallel
		{
#pragma omp for
			for (int i = 0; i < row; ++i)
			{
				for (int j = 0; j < col; ++j)
				{
					data[i*col+j] = assignfrom.data[i*col+j];
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

double& matrix::getdata(const int&i, const int&j){
	return data[i*col+j];
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

	double* d = data;

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

	data = Td;
	Td = nullptr;
	col = r;
	row = c;
	delete[] global;
	global = nullptr;
	delete[]d;
	d = nullptr;

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
	double*a = b.data;
	int& n = b.row;
	int&col = b.col;

	if (n != col) {
		throw std::invalid_argument("col and row should be the same!");
	}
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
				p = fabs(a[i*col+j]);
				if (p>d) { d = p; is[k] = i; js[k] = j; }
			}
		if (0.0 == d)
		{
			free(is); free(js);
			throw std::range_error("Can not be inversed !");
		}
		if (is[k] != k)
			for (j = 0; j <= n - 1; j++)
			{
				p = a[k*col+j];
				a[k*col+j] = a[is[k]*col+j];
				a[is[k]*col+j] = p;
			}
		if (js[k] != k)
			for (i = 0; i <= n - 1; i++)
			{
				p = a[i*col+k];
				a[i*col+k] = a[i*col+js[k]];
				a[i*col+js[k]] = p;
			}
		a[k*col+k] = 1.0 / a[k*col+k];
		for (j = 0; j <= n - 1; j++)
			if (j != k)
			{
				a[k*col+j] *= a[k*col+k];
			}
		for (i = 0; i <= n - 1; i++)
			if (i != k)
				for (j = 0; j <= n - 1; j++)
					if (j != k)
					{
						a[i*col+j] -= a[i*col+k] * a[k*col+j];
					}
		for (i = 0; i <= n - 1; i++)
			if (i != k)
			{
				a[i*col+k] = -a[i*col+k] * a[k*col+k];
			}
	}


	for (k = n - 1; k >= 0; k--)
	{
		if (js[k] != k)
			for (j = 0; j <= n - 1; j++)
			{
				p = a[k*col+j];
				a[k*col+j] = a[js[k]*col+j];
				a[js[k]*col+j] = p;
			}
		if (is[k] != k)
			for (i = 0; i <= n - 1; i++)
			{
				p = a[i*col+k];
				a[i*col+k] = a[i*col+is[k]];
				a[i*col+is[k]] = p;
			}
	}

	free(is); free(js);
	return std::move(b);
}


matrix operator*(const matrix& A, const matrix& B) {
	const char *programSouce =
		"__kernel                                         	\n"
		"void corefunc(const int M,                       	\n"
		"            const int N,                         	\n"
		"            const int P,                         	\n"
		"            __global double *a,                  	\n"
		"            __global double *b,                  	\n"
		"            __global double *c,                  	\n"
		"            __local double *bwrk)                \n"
		"{                                                \n"
		"  int k,j;                                       \n"
		"  int i = get_global_id(0);                      \n"
		"  int iloc = get_local_id(0);                    \n"
		"  int nloc = get_local_size(0);                  \n"
		"  double awrk[1000];                             \n"
		"  double tmp;                                    \n"
		"  if (i<N)                                       \n"
		"   {                                             \n"
		"		for (k=0;k<P;++k)                         \n"
		"			awrk[k]=a[i*P+k];                     \n"
		"		for (j=0;j<M;++j){                        \n"
		"			for (k=iloc;k<P;k=k+nloc)             \n"
		"				bwrk[k]=b[k*M+j];                 \n"
		"		barrier(CLK_LOCAL_MEM_FENCE);             \n"
		"			tmp=0;                                \n"
		"			for (k=0;k<P;++k)                     \n"
		"				tmp+=(awrk[k]*bwrk[k]);           \n"
		"			c[i*M+j]=tmp;                         \n"
		"		}                                         \n"
		"   }                                             \n"
		"}                                                \n"
		;
	int N = A.getrow();
	int P = A.getcol();

	int PB = B.getrow();
	int M = B.getcol();
	if (P != PB) {
		throw std::invalid_argument("Two matrices do not match the size!");
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

	double*a = A.data;
	double*b = B.data;

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
	err = clSetKernelArg(kernel, 6, sizeof(double)*P, NULL);

	size_t* global = new size_t[1];
	size_t* local = new size_t[1];
	global[0] = (size_t)N;
	local[0] = ((int)N + (int)4) / (int)5;

	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
	clFinish(queue);
	err = clEnqueueReadBuffer(queue, c_out, CL_TRUE, 0, sizeof(double)*szC, c, 0, NULL, NULL);

	matrix C(M, N, std::move(c));
	a = nullptr;
	b = nullptr;
	c = nullptr;
	delete[] global;

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

	double*a = A.data;
	double b = B;
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

	matrix C(row, col, std::move(c));
	c = nullptr;
	delete[] global;
	global = nullptr;
	a = nullptr;

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
		throw std::invalid_argument("Two matrices should have the same size!");
	}
	auto sz = col * row;

	cl_int err;
	cl_platform_id* pts = get_platforms();
	cl_platform_id platform = pts[0];
	cl_device_id* devices = get_devices(platform, CL_DEVICE_TYPE_GPU);
	cl_device_id device = devices[0];
	cl_context context = creat_context(platform, device, 1);
	cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

	double*a = A.data;
	double*b = B.data;
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

	matrix C (row, col, std::move(c));
	c = nullptr;
	delete[] global;
	global = nullptr;

	a = nullptr;
	b = nullptr;

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
		throw std::invalid_argument("two matrices should have the same size!");
	}
	auto sz = col * row;

	cl_int err;
	cl_platform_id* pts = get_platforms();
	cl_platform_id platform = pts[0];
	cl_device_id* devices = get_devices(platform, CL_DEVICE_TYPE_GPU);
	cl_device_id device = devices[0];
	cl_context context = creat_context(platform, device, 1);
	cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

	double*a = A.data;
	double*b = B.data;
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

	matrix C(row, col, std::move(c));
	c = nullptr;
	delete[] global;
	global = nullptr;

	a = nullptr;
	b = nullptr;

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
	int row = m.row;
	int col = m.col;
	for (int i = 0; i < row; i++)
	{
		os << " | ";
		for (int j = 0; j < col; j++)
		{
			char buf[32];
			double data = m.data[i*col+j];
			if (m.data[i*col+j] > -0.00001 && m.data[i*col+j] < 0.00001)
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

	auto sz = row * col;

	cl_int err;
	cl_platform_id* pts = get_platforms();
	cl_platform_id platform = pts[0];
	cl_device_id* devices = get_devices(platform, CL_DEVICE_TYPE_GPU);
	cl_device_id device = devices[0];
	cl_context context = creat_context(platform, device, 1);
	cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

	double*a = A.data;

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
	matrix C(row, 1, std::move(c));
	c = nullptr;
	delete[] global;
	global = nullptr;
	a=nullptr;
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

	double*a = miu.data;

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
	matrix C(row, col, c);
	c = nullptr;
	A = A - C;
	delete[] global;
	global = nullptr;
	a = nullptr;
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

cl_context& creat_context(const cl_platform_id& pt, const cl_device_id& devices, const int& numofdevcies) {
	cl_context_properties prop[] = { CL_CONTEXT_PLATFORM,(cl_context_properties)pt,0 };
	cl_context context = clCreateContext(prop, numofdevcies, &devices, NULL, NULL, NULL);
	return context;
}
#endif
