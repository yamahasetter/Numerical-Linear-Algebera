// these can be outside of #ifnotdef because they each have their own #ifnotdef
// inside of them
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <chrono> 
#include <cmath>
#include <math.h>
#include <vector>
#include <format>
#include <cassert>
#include <iomanip>
#include <iterator>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <tuple>
#include <string>

using namespace std;
using namespace std::chrono;

// this runs the first time the driver imports and will not run if the
// 'define' statement has been run before and the contents of the 'define'
// statement remain unchanged
#ifndef HW5_H
#define HW5_H

double trace(vector<vector<double> > matrixA, int m);
double twonorm(vector<double> vectorA);
void dispmat(vector<vector<double> > matrixA);
vector<vector<double> > dat_to_mat(string filename);
tuple<vector<vector<double> >, vector<vector<double> > > GE(vector<vector<double> > matrixA, vector<vector<double> > matrixB);
vector<vector<double> > backsub(vector<vector<double> > matrixA, vector<vector<double> > matrixB);
tuple<vector<vector<double> >, vector<vector<double> > > partialpivotpair(vector<vector<double> > matrixA, vector<vector<double> > matrixB);
vector<vector<double> > partialpivot(vector<vector<double> > matrixA);
double error(vector<vector<double> > matrixA, vector<vector<double> > matrixB, vector<vector<double> > X);
tuple<vector<vector<double> >, vector<int> > LUdecomp(vector<vector<double> > matrixA);
bool ismachzero(double test);
vector<vector<double> > permute(vector<vector<double> > matrix, vector<int> s);
tuple<vector<vector<double> >, vector<vector<double> > > LUbacksub(vector<vector<double> > matrixA, vector<vector<double> > matrixB, vector<int> s);
vector<vector<double> > cholesky(vector<vector<double> > A);
vector<vector<double> > choleskybacksub(vector<vector<double> > A, vector<vector<double> > B);
vector<vector<double> > vandermonde(vector<double> x, int degree);
tuple<vector<double>, vector<double> > getvectors(string filename, int n);
void write_data_to_file(const string filepath, const vector<vector<double>>& data);
vector<vector<double> > matrixmult(vector<vector<double> > A, vector<vector<double> > B);
vector<vector<double> > transpose(vector<vector<double> > A);
tuple<vector<vector<double> >, vector<vector<double> > > HHVR(vector<vector<double> > A);
vector<vector<double> > outerproduct(vector<double> x);
tuple<vector<vector<double> >, vector<vector<double> > > QRtohat(vector<vector<double> > Q, vector<vector<double> > R);
vector<vector<double> > HHQTB(vector<vector<double> > V,vector<vector<double> > B);
double frobnorm(vector<vector<double> > A);
vector<vector<double> > matdiff(vector<vector<double> > A, vector<vector<double> > B);
vector<vector<double> > identity(int m);
vector<vector<double> > QfromV(vector<vector<double> > V);
tuple<vector<vector<double> >, vector<vector<double> > > hessenberg(vector<vector<double> > A);
double diagdiff(vector<vector<double> > A, vector<vector<double> > B);
vector<vector<double> > iterQR(vector<vector<double> > A);
vector<vector<double> > iterQRshifted(vector<vector<double> > A, double shift);
vector<vector<double> > inviter(vector<vector<double> > A, double shift);


#endif