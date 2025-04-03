#include "../include/hw5.h"
#include <iostream>
#include <vector>

int main(){
	// read the data file -> created from the matrix (1) in HW5
	string Afilename = "data/Amat.dat";
	vector<vector<double> > A = dat_to_mat(Afilename);
	cout << "On start A:" << endl;
	dispmat(A);

	tuple<vector<vector<double> >, vector<vector<double> > > AVtuple = hessenberg(A);
	// tuple<vector<vector<double> >, vector<vector<double> > > AVtuple = HHVR(A);
	A = get<0>(AVtuple);
	vector<vector<double> > V = get<1>(AVtuple);

	cout << "Tridiagonal A:" << endl;
	dispmat(A);

	// make a new A
	string Bfilename = "data/Bmat.dat";
	A = dat_to_mat(Bfilename);
	cout << "On start next A:" << endl;
	dispmat(A);


	cout << "Starting iterative QR to find eigenvalues..." << endl;
	vector<vector<double> > D = iterQR(A);
	cout << "Succesfully Calculated D without shift:" << endl;
	dispmat(D);

	D = iterQRshifted(A, .005);
	cout << "Succesfully Calculated D with shift:" << endl;
	dispmat(D);

	// make yet another A...
	string Cfilename = "data/Cmat.dat";
	A = dat_to_mat(Cfilename);
	cout << "On start next A:" << endl;
	dispmat(A);


	// choose your shift
	// double shift = -8.0286;
	// double shift =7.9329;
	// double shift =5.6689;
	double shift =-1.5732;
	cout << "solving for eigenvector..." << endl;
	vector<vector<double> > X = inviter(A,shift);

	cout << "=== Succesfully found eigenvector: ===" << endl;
	for (unsigned long i = 0; i < X.size(); ++i){
		cout << X[i][0] << ", ";
	}
	cout << endl;

}
























