#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <math.h>
#include <vector>
#include <format>
#include <cassert>
#include <iomanip>
#include <iterator>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <tuple>

using namespace std;

/* 
* A function that takes in a matrix and its row dimension and 
* returns a human-readable matrix to the screen
*/
void dispmat(vector<vector<double> > matrixA){
    int rows_m = matrixA.size();
    // iterate over the rows and print their contents
    ostream_iterator<double> out (cout, ",    ");
    for (int row=0; row<rows_m; row++){
        copy ( matrixA[row].begin(), matrixA[row].end(), out );
        cout << endl;
    }
}


/*
* A helper function to swap rows
*/
vector<vector<double> > swaprows(vector<vector<double> > matrixA, int row1, int row2, bool verbose = false){
    // get length of A
    int Acols = matrixA[0].size();
    if(verbose){
        // show before rowswap
        cout << "Before row swap:" << endl;
        dispmat(matrixA);
    }
    for (int j=0; j<Acols; j++){
        double hold = matrixA[row1][j];
        matrixA[row1][j] = matrixA[row2][j];
        matrixA[row2][j] = hold;
    }
    if (verbose){
        // show after rowswap
        cout << "After row swap:" << endl;
        dispmat(matrixA);
    }
    return matrixA;
}


/*
* A helper function to partial pivot
*/
vector<vector<double> > partialpivot(vector<vector<double> > matrixA){
    // get the size of A
    int Acols = matrixA[0].size();
    int Arows = matrixA.size();

    // show matrix before
    dispmat(matrixA);

    // iterate over the columns
    for (int j=0; j<Acols; j++){
        // check to see if LT rows have larger element that a_jj
        int row = j;
        for (int i=row+1; i<Arows; i++){
            if (matrixA[i][j] >= matrixA[row][j]){
                row = i;
            } 
            else {
                // do nothing
            }
        }
        if (row != j){
            // perform row swap so that a_jj = a_ij where a_ij is bigger than old a_jj
            matrixA = swaprows(matrixA, j, row); 
        }

    }
    cout << "Partial Pivoting for Matrix Complete." << endl;
    // show matrix after
    dispmat(matrixA);
    return matrixA;
}


/*
* A helper function to partial pivot a pair of matrices AND IT IS ACTUALLY USELESS SO THAT'S AWESOME
*/
tuple<vector<vector<double> >, vector<vector<double> > > partialpivotpair(vector<vector<double> > matrixA, vector<vector<double> > matrixB){
    // get the size of A
    int Acols = matrixA[0].size();
    int Arows = matrixA.size();

    // show matrices before
    dispmat(matrixA);
    dispmat(matrixB);

    // iterate over the columns
    for (int j=0; j<Acols; j++){
        // check to see if LT rows have larger element that a_jj
        int row = j;
        for (int i=row+1; i<Arows; i++){
            if (matrixA[i][j] >= matrixA[row][j]){
                row = i;
            } 
            else {
                // do nothing
            }
        }
        if (row != j){
            // perform row swap so that a_jj = a_ij where a_ij is bigger than old a_jj
            matrixA = swaprows(matrixA, j, row);
            // do same on matrix B
            matrixB = swaprows(matrixB, j, row); 
        }

    }
    cout << "Partial Pivoting for Matrices Complete." << endl;
    // show matrices after
    dispmat(matrixA);
    dispmat(matrixB);
    return {matrixA, matrixB};
}

/* A function that takes in 2 arguments: an mxm square matrix,
* and the value of m. The function will return the trace of A. 
*/
double trace(vector<vector<double> > matrixA, int m){
    double sum=0;
    for (int i=0; i<m; i++){
        sum = sum + matrixA[i][i];
    }
    cout << "Trace Calculated Successfully:" << endl;
    cout<<sum<<endl;
    return sum;
}

/* A function that takes in a vector and its length
* and return the 2-norm
*/

double twonorm(vector<double> vectorA){
    int n = vectorA.size();
    double norm=0.0;
    for (int i=0; i<n; i++){
        double index = vectorA[i];
        norm = norm + pow(index, 2.0);
    }
    //cout << "Successfully Calculated 2-Norm:" << endl;
    return sqrt(norm);
}

/* A function that takes in an array and its length
* and return the 2-norm
*/

double twonormarray(double vectorA[], int n){
    double norm=0.0;
    for (int i=0; i<n; i++){
        double index = vectorA[i];
        norm = norm + pow(index, 2.0);
    }
    //cout << "Successfully Calculated 2-Norm." << endl;
    return sqrt(norm);
}

/*
* A function that takes in a filename for a .dat file of matrix
* values and returns a cpp "matrix"
*/
vector<vector<double> > dat_to_mat(string filename){
    // open the file
    ifstream file;
    file.open(filename);

    // get rows and cols
    int rows, cols;
    file >> rows >> cols;

    // make an empty matrix
    vector<vector<double> > matrixA(rows, vector<double> (cols,0));

    // iterate thru the file and build the matrix
    int i=0, j=0;
    for (int iter=0; !file.eof(); iter++){        
        // notice that we start at indices [0][0] and a row ends
        // when the second index == cols.
        // if the col index is < the # cols, continue to populate that row
        if (j<cols){
            file >> matrixA[i][j];
            j++;
        }
        // else when col index == cols, we need to start a new row and reset the col index
        else if (j>=cols){
            j = 0;
            i++;
            file >> matrixA[i][j];
            j++;
        }
        }
        
    cout << "Matrix Generation Successful:" << endl;
    return matrixA;
}


/*
* Given a vector, generate a matrix
*/

vector<vector<double> > outerproduct(vector<double> x){
    int size = x.size();

    // build empty matrix
    vector<vector<double> > X(size, vector<double> (size,0));

    // compute x x^T 
    for (int i = 0; i < size; ++i){
        for (int j = 0; j < size; ++j){
            X[i][j] = x[i]*x[j];
        }
    }
    return X;
}




/*
* Given a file with n tuples (x,y) we return n-long vectors x, y that contain the points
*/
tuple<vector<double>, vector<double> > getvectors(string filename, int n){
    // open the file
    ifstream file;
    file.open(filename);

    // initialize the vectors
    vector<double> x(n);
    vector<double> y(n);

    for (int i = 0; i < n; ++i){
        // assign x
        file >> x[i];
        // assign y
        file >> y[i];
    }
    cout << "Vector Generation Successful." << endl;
    return {x,y};
}


/*
* Returns 1 (True) if a number is machine zero and 0 (False) if a number is not machine zero
*/
bool ismachzero(double test){
    double tolerance = 1.0e-6;
    if(fabs(test)<tolerance){
        return true;
    }
    else{
        return false;
    }
}



/*
* Given matrices A and B, use Gaussian Elimination to get A Upper-Triangular. For every operation
* on A, perform the same operation on B and X.
* 
* We are trying to solve the system AX=B, where all elements are matrices.
*
* The general process is as follows:
* For each row in A, find A[i][j]/A[j][j] where i<j.
* If A[j][j] = 0, then we cannot proceed.
* Scale row j: A[j]*(A[i][j]/A[j][j]).
* For each element in row i: A[i][k] - A[j][k]*(A[i][j]/A[j][j]) for k in cols of A.
* Perform the same operation exactly on the corresponding row of B.
*/

tuple<vector<vector<double> >, vector<vector<double> > > GE(vector<vector<double> > matrixA, vector<vector<double> > matrixB){
    // We extract the dims of each matrix using the size() function of the vector class
    int Arows = matrixA.size();
    int Acols = matrixA[0].size();
    // int Brows = matrixB.size(); --> this should be the same as A
    int Bcols = matrixB[0].size();

    // we print originals to screen:
    cout << "Matrix A before Gaussian Elimination w/ Partial Pivoting: " << endl;
    dispmat(matrixA);
    cout << "Matrix B before Gaussian Elimination w/ Partial Pivoting: " << endl;
    dispmat(matrixB);

    // We loop over the columns of A
    for (int j=0; j<Acols; j++){
        // we swap rows so that the largest entry in this jth column is along the diagonal
        int row = j;
        for (int i=j+1; i<Arows; i++){
            if (abs(matrixA[i][j]) > abs(matrixA[row][j])){
                row = i;
            } 
            else {
                // do nothing
            }
        }
        if (row != j){
            // perform row swap so that a_jj = a_ij where a_ij is bigger than old a_jj
            matrixA = swaprows(matrixA, j, row); 
            // do the same for matrix B:
            matrixB = swaprows(matrixB, j, row); 
        }

        // we check the diagonal entry a[j][j] is not machine zero
        double ajj = matrixA[j][j];
        if (ismachzero(ajj)){
            cout << "Zero found along diagonal: Matrix is Singular!" << endl;
            exit(0);
        }
        // if it's not 0 we loop over the rows below A[][j]
        else{
            for (int i=j+1; i<Arows; i++){
                // Calculate A[i][j]/A[j][j]
                double rescale = matrixA[i][j]/ajj;
                // for each element in the i'th row, subtract the corresponding element
                // in the rescaled jth row
                for (int Aindex=0; Aindex<Acols; Aindex++){
                    matrixA[i][Aindex] = matrixA[i][Aindex] - matrixA[j][Aindex]*rescale;
                }
                // for matrix B
                for (int Bindex =0; Bindex<Bcols; Bindex++){
                    matrixB[i][Bindex] = matrixB[i][Bindex] - matrixB[j][Bindex]*rescale;
                }
            }
        }
    }
    // we print uper-triangular A to screen with modified B:
    cout << "Matrix A after Gaussian Elimination w/ Partial Pivoting: " << endl;
    dispmat(matrixA);
    cout << "Matrix B after Gaussian Elimination w/ Partial Pivoting: " << endl;
    dispmat(matrixB);

    // output a tuple with the new A and B values
    return {matrixA, matrixB};
}



/*
* Given UT matrix A and another matrix not neccessarily UT, B, backsolve for X
* in the equation AX=B and return X
*
* The algorithm is as follows:
* for each column of X, select the same of column of B and solve for Ax=b.
* Recall that the last entry of x = last entry of b / bottom right of A (a_mm)
* Backsolve from there using:
* X[i][j] = (B[i][j]-sum_{k=0}^{Acols} A[i][k]*X[k][j])/A[i][i]
*/

vector<vector<double> > backsub(vector<vector<double> > matrixA, vector<vector<double> > matrixB){
    // we know X must be the same size as matrixB, so we get those dims. Also, Arows = Brows always
    int Brows = matrixB.size()-1; // -1 since indexing starts at 0 and we want to index the last element
    int Bcols = matrixB[0].size();
    int Acols = matrixA[0].size();

    // if the bottom right entry of A is 0, then A is singular:
    if (ismachzero(matrixA[Brows][Brows])){
        cout << "Matrix is Singular!" << endl;
        exit(0);
    }

    // make an empty X matrix with same size as B
    vector<vector<double> > X(Brows+1, vector<double> (Bcols,0));

    // start on the first column of X and B
    for (int j=0; j<Bcols; j++){
        // set the last element of the j'th column in X via:
        double xmm = matrixB[Brows][j]/matrixA[Brows][Brows];
        X[Brows][j] = xmm;
        // iterate over the rows of A from second-to-bottom to top
        for (int i=Brows-1; i>-1; i--){
            // if the diagonal element is 0 we have a singular matrix
            if (ismachzero(matrixA[i][i])){
                cout << "Matrix is Singular!" << endl;
                exit(0);
            }
            // backsolve for X[Brows-N][j] case and keep track of b-x/a terms with weight
            double weight = 0.0;
            // iterate over the columns of A[i] for col>row
            for (int col=i+1; col<Acols; col++){
                // calculate A[i][col]*X[col][j] and store
                weight = weight + matrixA[i][col]*X[col][j];
            }
            // when the loop terminates at the end of the ith row of A we set the value of X[i][j]
            // we do a quick check to make 0's be 0's:
            if(ismachzero((matrixB[i][j]-weight)/matrixA[i][i])){
                X[i][j] = 0.0;
            }
            else{
                X[i][j] = (matrixB[i][j]-weight)/matrixA[i][i];
            }
        }
    }
    cout << "Backsubstitution of X complete." << endl;
    return X;
}


/*
* Calculate error of AX=B and return array of 2 normed errors
*/
double error(vector<vector<double> > matrixA, vector<vector<double> > matrixB, vector<vector<double> > X){
    // get the shape of A, B and X
    int Brows = matrixB.size();
    int Bcols = matrixB[0].size();
    int Acols = matrixA[0].size();

    // initialize error array
    double error[Bcols];

    // we first generate a column vector via Ax then we iteratively compute Ax - b
    // get the column vectors from X and B
    for (int j=0; j<Bcols; j++){
        // initialize a vector to hold Ax vector
        double y[Brows];
        // loop over rows of A
        for (int row = 0; row<Brows; row++){
            // record the dot product of each a*x^T
            double dot = 0.0;
            // iterate over the columns of A and rows of x to find Ax
            for (int col=0; col<Acols; col++){
                dot = dot + matrixA[row][col]*X[col][j];
            }
            // record Ax
            y[row] = dot;
        }
        // find Ax-b
        for (int index=0; index<Brows; index++){
            // round to 0 if needed and reuse the same array
            // if (ismachzero(y[index] - matrixB[index][j])){
            //     y[index] = 0.0;
            // }
            // else{
            //     y[index] = y[index] - matrixB[index][j];
            // }
            y[index] = y[index] - matrixB[index][j];
        }
        // record 2 norm error
        error[j] = twonormarray(y, Brows);
    }
    // return the completed error array
    cout << "Error Calculated Successfully:" << endl;
    for (int i=0; i<Bcols; i++) {
        cout << error[i] << ", ";
    }
    cout << endl;
    return 0;
}


/*
* LU factorization by GE with partial pivoting
*/
tuple<vector<vector<double> >, vector<int> > LUdecomp(vector<vector<double> > matrixA){
    // get the size of A (A is square)
    int size = matrixA.size();

    // make the permutation vector s:
    vector<int> s(size);
    // set the current order of the rows
    for(int i=0; i<size; i++){
        s[i] = i;
    }

    for (int j=0; j<size; j++){
        // --- start row swap ---
        // we swap rows so that the largest entry in this jth column is along the diagonal
        int row = j;
        for (int i=j+1; i<size; i++){
            if (abs(matrixA[i][j]) > abs(matrixA[row][j])){
                row = i;
            } 
            else {
                // do nothing
            }
        }
        if (row != j){
            // perform row swap so that a_jj = a_ij where a_ij is bigger than old a_jj
            matrixA = swaprows(matrixA, j, row); 
            // record the permutation in the permutation vector
            int temp = s[j];
            s[j]=s[row];
            s[row]=temp;
        }
        
        // we check the diagonal entry a[j][j] != 0
        double ajj = matrixA[j][j];
        if (ismachzero(ajj)){
            cout << "Cannot Factorize Matrix!" << endl;
            exit(0);
        }

        // if it's not 0 we loop over the rows below A[][j] to build LU
        else{
            for (int i=j+1; i<size; i++){
                // Calculate A[i][j]/A[j][j]
                matrixA[i][j] = matrixA[i][j]/ajj;
                // for each element in the i'th row, subtract the corresponding element
                // in the rescaled jth row
                for (int k=j+1; k<size; k++){
                    matrixA[i][k] = matrixA[i][k] - matrixA[j][k]*matrixA[i][j];
                }
            }
        }

    }
    cout << "Matrix Factorization Complete." << endl;
    // we print uper-triangular A to screen with derived L and U:
    cout << "Matrix A after Factorization: " << endl;
    dispmat(matrixA);
    cout << "Permutation Vector: " << endl;
    for (int i=0; i<size; i++) {
        cout << s[i] << ", ";
    }
    cout << endl;

    // output a tuple with the matrices
    return {matrixA, s};
}


/*
* Given a matrix and a permutation vector return a permuted matrix
*/
vector<vector<double> > permute(vector<vector<double> > matrix, vector<int> s){
    // get the dims of the matrix
    int rows=matrix.size();
    int cols=matrix[0].size();
    // make a return matrix of the same size
    vector<vector<double> > matrixout(rows, vector<double> (cols,0));
    // iterate over the permutation vector and generate the permuted matrix
    for (int i=0; i<rows; i++){
        for (int j=0; j<cols; j++){
            matrixout[i][j]= matrix[s[i]][j];
        }
    }
    return matrixout;
}



/*
* Given LU factorized A with permutation vector S, we take L and U from A and backsolve for a matrix X
* We also return the manipulated B matrix
*/

tuple<vector<vector<double> >, vector<vector<double> > > LUbacksub(vector<vector<double> > matrixA, vector<vector<double> > matrixB, vector<int> s){
    // we know X must be the same size as matrixB, so we get those dims. Also, Arows = Brows always
    int Brows = matrixB.size()-1; // -1 since indexing starts at 0 and we want to index the last element
    int Bcols = matrixB[0].size();
    int Acols = matrixA[0].size();

    // if the bottom right entry of A is 0, then A is singular:
    if (ismachzero(matrixA[Brows][Brows])){
        cout << "Matrix is Singular!" << endl;
        exit(0);
    }

    // make an empty X matrix with same size as B
    vector<vector<double> > X(Brows+1, vector<double> (Bcols,0));
    // Permute B according to s
    matrixB = permute(matrixB, s);

    // start on the first column of X and B
    for (int j=0; j<Bcols; j++){
        // make the vector y to hold the forward substitution y=L^{-1}Pb_{j}
        vector<double> y(Brows+1);
        for (int i=0; i<Brows+1; i++){
            y[i]=matrixB[i][j];
        }

        // loop through the columns of A and solve for y
        for (int col=0; col<Acols-1; col++){
            // loop through the LT rows of A
            for(int row=col+1; row<Brows+1; row++){
                y[row]=y[row]-y[col]*matrixA[row][col];
            }
        }
        // backsub Ux=y
        // start on the bottom row
        for(int i=Brows; i>-1; i--){
            // check if a_{ii} = 0
            if (ismachzero(matrixA[i][i])){
                cout << "MATRIX IS SINGULAR!" << endl;
                exit(0);
            }
            // otherwise we continue to backsolve for x
            else{
                double sum = 0.0;
                // loop over UT columns -> note that this will not activate on the first pass of the
                // outer loop since k = m + 1 > m. So x_mj will be (y_i - 0)/a_ii 
                for(int k=i+1; k<Brows + 1; k++){
                    sum = sum + matrixA[i][k]*X[k][j];
                }
                if (ismachzero((y[i]-sum)/matrixA[i][i])){
                    X[i][j]=0;
                }
                else{
                    X[i][j]=(y[i]-sum)/matrixA[i][i];
                }
            }
        }
        // when these inner loops terminate, we have solved a column of X
        //cout << "Solved for column " << j << " of X" << endl; 
    }
    cout << "Backsubstitution of X complete:" << endl;
    dispmat(X);
    return {X, matrixB};
}


/*
* Returns Cholesky Factorization of Square Matrix:
* Given a square matrix A (that is positive-definite and Hermitian)
* we can factor A into L and U such that A = LL* = U*U where ujj = ljj >0
*
* The algorithm is as follows:
*   - Loop over the columns of A with j 
*       -Calculate the new diagonal element:
*           - loop over the upper cols with k
*               - set ajj = ajj - ajk * akj
*           - set ajj = sqrt(ajj)
*       -Calculate the elements below the diagonal
*           - loop over the upper rows with i
*               - loop over the upper columns with k
*                   set aij = aij - aik * akj
*               - set aij = aij/ajj
*/


vector<vector<double> > cholesky(vector<vector<double> > A){
    // get the dims of the matrix
    int rows=A.size();
    int cols=A[0].size();

    // loop over the columns
    for (int j = 0; j<cols; j++){
        // calculate the new diagonal element
        for (int k = 0; k<j; k++){
            A[j][j] = A[j][j] - pow(A[j][k],2); // THIS IS THE BUG! P79 1.10.4 IN TEXTBOOK MIGHT WORK INSTEAD
        }
        // set the diagonal entry
        if (A[j][j]<=0.0){
            cout << "Cannot Factorize Matrix!" << endl;
            cout << "MATRIX NOT PD!" << endl;
            exit(0);
        }
        else{
            A[j][j] = sqrt(A[j][j]);
        }

        // calculate the elements below the diagonal
        for (int i = j+1; i < rows; i++){
            for (int k=0; k<j; k++){
                A[i][j] = A[i][j] - A[i][k]*A[j][k];
            }
            // set the new entry
            A[i][j] = A[i][j] / A[j][j];
        }
    }
    // return the matrix
    cout << "Cholesky Factorization Complete: " << endl;
    dispmat(A);
    return A;
}


/*
* Cholesky Backsubstitution
* Performs a forward step via Ly = b
* Performs a backsub step via L^* x = y
* The result is the solution x
* - Iterate over each solution vector in your solution matrix B
*    - Get the forward substitution:
*        - Ly = b
*    - Get the backward substitution:
*        - L^* x = y
*  
* On return we get the solution matrix X
*/

vector<vector<double> > choleskybacksub(vector<vector<double> > A, vector<vector<double> > B){
    // get the dims of the matrix
    int Brows = B.size()-1; // -1 since indexing starts at 0 and we want to index the last element
    int Bcols = B[0].size();
    int Acols = A[0].size();
    // make an empty X matrix with same size as B
    vector<vector<double> > X(Brows+1, vector<double> (Bcols,0));
    // loop over the columns of B
    for (int bcol = 0; bcol < Bcols; bcol++){
        // *-*-* Forward Sub:
        // loop over the rows of A
        // make the vector y to hold the forward substitution Ly=b
        vector<double> y(Brows+1);
        for (int i = 0; i < Brows+1; i++){
            double sum = B[i][bcol];
            // loop over upper cols
            for (int j = 0; j<i; j++){
                sum = sum - y[j]*A[i][j];
            }
            y[i] = sum/A[i][i];
        }
        // *-*-* Backward Sub:
        // loop over rows bottom-to-top
        for (int i = Brows; i>-1; i--){
            // check if Lii is 0
            if (ismachzero(A[i][i])){
                cout << "MATRIX IS SINGULAR!" << endl;
                exit(0);
            }
            else{
                // backsub L* y=x
                for (int k = i+1; k <Acols; k++){
                    y[i] = y[i] - A[k][i]*X[k][bcol];
                }
                X[i][bcol] = y[i]/A[i][i];
            }  
        }
    }
    cout << "Cholesky Backsubstitution Complete." << endl;
    cout << "X:" << endl;
    dispmat(X);
    return X;
}


/*
* Given dimensions a vector we will generate a square Vandermonde matrix
*/
vector<vector<double> > vandermonde(vector<double> x, int degree){
    // initialize to ones
    vector<vector<double> > V(x.size(), vector<double> (degree+1, 1));
    
    // each row is a polynomial: 1 + x + x^2 + ... x^(n-1)
    for (unsigned long i = 0; i < x.size(); ++i){
        // we skip the first column because it's already 1
        for (int j = 1; j < degree+1; ++j){
            // go to the corresponding row of x and raise it to the power j
            // if that element is nonzero keep it, otherwise set it to zero:
            if(ismachzero(pow(x[i],j))){
                V[i][j] = 0.0;
            }
            else{
                V[i][j] = pow(x[i],j);    
            }
        }
    }
    cout << "Vandermonde Matrix Generation Successful:" << endl;
    dispmat(V);
    return V;
}

/*
* Helper function to write data to a file (written by ChatGPT)
*/
void write_data_to_file(const string filepath, const vector<vector<double>>& data){
    ofstream outfile(filepath);
    if (outfile.is_open()) {
        for (const auto& row : data) {
            for (const auto& elem : row) {
                outfile << elem << " ";
            }
            outfile << "\n";
        }
        outfile.close();
        cout << "Data has been written to " << filepath << endl;
    } else {
        cout << "Unable to open file " << filepath << " for writing" << endl;
    }
}


// transpose of a matrix
vector<vector<double> > transpose(vector<vector<double> > A){
    // make empty matrix
    vector<vector<double> > AT(A[0].size(), vector<double> (A.size(),0));

    for (unsigned long i = 0; i<A.size();i++){
        for (unsigned long j = 0; j<A[0].size(); j++){
            AT[j][i] = A[i][j];
        }
    }
    cout << "Transpose Calculated Successfully."<< endl;
    return AT;
}

// multiply two matrices
vector<vector<double> > matrixmult(vector<vector<double> > A, vector<vector<double> > B){
    // make output
    vector<vector<double> > C(A.size(), vector<double> (B[0].size(),0));

    // iterate over the rows of A
    for(unsigned long i=0; i<A.size(); i++){
        // and the columns of B
        for(unsigned long j=0; j<B[0].size(); j++){
            // now record the dot product of ai bj
            double sum = 0.0;
            for(unsigned long k=0; k<A[0].size(); k++){ //A[0].size = B[0].size
                sum = sum + A[i][k]*B[k][j];
            }
            // set the entry in C
            if(ismachzero(sum)){
                C[i][j] = 0.0;
            }
            else{
                C[i][j] = sum;
            }
        }
    }
    cout << "Matrix Product Calculated Successfully" << endl;
    return C;
}


/*
* Householder QR factorization
*/

tuple<vector<vector<double> >, vector<vector<double> > > HHVR(vector<vector<double> > A){
    // get the dims of the matrix
    int rows=A.size();
    int cols=A[0].size();
    // make V to store vj
    vector<vector<double> > V(rows, vector<double> (cols,0));

    // loop over columns of A
    for (int j = 0; j < cols; j++){
        // make vector s and v
        vector<double> s(rows);
        vector<double> v(rows,0);

        // compute entries of s 
        double sum = 0.0;
        //iterate over the columns of A
        for (int i = j; i < rows; i++){
            sum = sum + pow(A[i][j],2);
        }
        // this trick: (x > 0) - (x < 0) yeilds +1, -1, 0 if sign(x)=+,-,0
        s[j] = ((A[j][j] > 0) - (A[j][j] < 0)) * sqrt(sum);

        // set the values of vj
        v[j] = A[j][j] + s[j];
        for (int i = j+1; i < rows; i++){
            v[i] = A[i][j];
        }
        // get the norm of vj
        double vnorm = twonorm(v);

        // rescale vj
        for (int i = 0; i < rows; i++){
            if (ismachzero(v[i]/vnorm)){
                v[i]=0.0;
            }
            else{
                v[i]=v[i]/vnorm;
            }   
        }
        // record vj in V
        for(int i=0; i<rows; i++){
            V[i][j]=v[i];
        }

        // compute vv^T
        vector<vector<double> > vvt=outerproduct(v);

        // compute vvT A
        vector<vector<double> > vvtA=matrixmult(vvt,A);

        // update: A = A - 2HA
        for (int i = 0; i < rows; i++){
            for (int j2 = 0; j2 < cols; j2++){
                if (ismachzero(A[i][j2] - 2*vvtA[i][j2])){
                    A[i][j2] = 0.0;
                }
                else{
                    A[i][j2]= A[i][j2] - 2*vvtA[i][j2];    
                }
            }
        }
    }
    return {A, V};
}


/*
* given a list of vectors V generate the householder reflectors and build Q^T bi -> B
* this takes v, does b - 2v * v^T b and makes this a columnd of B
* we return the modified B matrix
*/ 
vector<vector<double> > HHQTB(vector<vector<double> > V,vector<vector<double> > B){
    // get size of V matrix
    int vrows=V.size();
    int vcols=V[0].size();
    // get size of B matrix
    // int brows=B.size(); --> should be the same as vrows?
    int bcols=B[0].size();

    // iterate over columns of B
    // for each v_j, we need to find b - 2 v_j * v_j^T b / ||v_j||
    for (int b=0; b<bcols; b++){
        // do H1 H2 H3 ... Hn b
        for (int v = 0; v<vcols; v++){
            // get 2 norm squared of v_j
            double vnorm = 0.0;
            for (int i =0; i<vrows; i++){
                vnorm = vnorm + pow(V[i][v],2);
            }
            // get 2 v^T b
            double vtb = 0.0;
            for (int i = 0; i<vrows; i++){
                vtb = vtb + 2*V[i][v]*B[i][b];
            }
            // get b_i - v_i * 2vTb / ||v||^2
            for (int i=0; i<vrows; i++){
                // Hk b[i] = B[i][b] - V[i][v]*vtb;
                B[i][b] = B[i][b] - (V[i][v]*vtb)/vnorm;
            }
        }
    }
    return B;
}


/*
* Given Q and R, return Qhat and Rhat
*
* We find the number of nonzero rows of R and set it to n
* Then we cut the last m-n rows of R off and the last m-n columns of Q off
* the result is Qhat and Rhat
*/

tuple<vector<vector<double> >, vector<vector<double> > > QRtohat(vector<vector<double> > Q, vector<vector<double> > R){
    // get sizes
    int rrows=R.size();
    int rcols=R[0].size();
    //int qrows=Q.size();
    //int qcols=Q[0].size();

    // number of nonzero rows
    int n=0;

    // iterate over rows looking for first zero row
    for (int row =0; row<rrows; row++){
        // record the entries
        double sum = 0.0;
        for (int j = 0; j < rcols; j++){
            sum = sum + R[row][j];
        }
        // if the sum is 0 then we have found the first n nonzero rows where n is a bound: [0,n)
        if (ismachzero(sum)){
            n = row;
            break;
        }
    }

    // resize accordingly
    R.resize(n);
    Q.resize(n);
    // make matrices 
    // vector<vector<double> > Qhat(qrows, vector<double> (n-1,0));
    // vector<vector<double> > Rhat(n-1, vector<double> (rcols,0));
    // cout << "Qhat" << endl;
    // dispmat(Q);
    // cout << "Rhat" << endl;
    // dispmat(R);
    return {Q,R};
}





/*
* Frobenius Norm
* Given a real matrix A, this function will transpose it, take the product A^T A
* compute the trace of A^T A and then sqrt the trace
*/

// double frobnorm(vector<vector<double> > A){
//     // transpose A
//     vector<vector<double> > AT = transpose(A);
//     // overwrite AT with matrix product (saves og matrix)
//     AT = matrixmult(AT,A);
//     // trace
//     double norm = trace(AT,AT.size());
//     return sqrt(norm);
// }

// this is the more generalized frobenius norm but suprisingly it gives the same answer as the old one :D
double frobnorm(vector<vector<double> > A){
    int rows = A.size();
    int cols = A[0].size();

    double sum = 0.0;
    for (int i = 0; i < rows; ++i){
        for (int j = 0; j < cols; ++j){
            sum += pow(A[i][j],2);
        }
    }
    return sqrt(sum);
}



/*
* Difference of matrices
* Given a real matrix A, B this function gets the difference
*/
vector<vector<double> > matdiff(vector<vector<double> > A, vector<vector<double> > B){
    // A and B are same size
    int rows = A.size();
    int cols = A[0].size();

    // make new matrix
    vector<vector<double> > diff(rows, vector<double>(cols,0));

    for (int i = 0; i < rows; ++i){
        for (int j = 0; j < cols; ++j){
            diff[i][j] = A[i][j]-B[i][j];
        }
    }
    return diff;
}


// Make an identity matrix
vector<vector<double> > identity(int m){
    vector<vector<double> > I(m, vector<double>(m,0));
    for (int i = 0; i < m; ++i){
        I[i][i] = 1;
    }
    return I;
}


/*
* Given a set of householder vectors V, generate Q
* For each vector V, we generate a householder matrix H_i
* Then we compute the next one H_i+1 and compute H_i * H_(i+1) = placeholder
*/
vector<vector<double> > QfromV(vector<vector<double> > V){
    // get sizes of V
    int rows = V.size();
    int cols = V[0].size();

    // build Q
    vector<vector<double> > Q(rows, vector<double>(rows,0));

    // build H1 and update Q
    vector<vector<double> > H(rows, vector<double>(rows,0));

    // build a placeholder v
    vector<double> v(rows,0); // v1
    for (int i = 0; i < rows; ++i){
        v[i] = V[i][0];
    }
    H = outerproduct(v); // v1v1^T
    // cout << "vv^T" << endl;
    // dispmat(H);

    // norm
    double norm = twonorm(v);
    // do I - 2 vv^T/|v|
    for (int i = 0; i < rows; ++i){
        for (int j = 0; j < rows; ++j){
            if(i==j){
                H[i][j] = 1 - 2*H[i][j]/norm;
            }
            else{
                H[i][j] = -2*H[i][j]/norm;
            }
        }
    }
    // now we have H1 and can set it as Q
    Q = H;

    // now we build H_i and update Q
    for (int col = 1; col < cols; ++col){ // 1 bc we already did the first col
        // make vi
        for (int i = 0; i < rows; ++i){
            v[i] = V[i][col];
        }
        H = outerproduct(v); // vivi^T
        norm = twonorm(v);
        // do I - 2 vivi^T/|vi|
        for (int i = 0; i < rows; ++i){
            for (int j = 0; j < rows; ++j){
                if(i==j){
                    H[i][j] = 1 - 2*H[i][j]/norm;
                }
                else{
                    H[i][j] = -2*H[i][j]/norm;
                }
            }
        }
        // now we have Hi and Q = Q*Hi
        Q = matrixmult(Q,H);
    }
    // we have found Q
    return Q;
}



/*
* Hessenberg Reduction
* Given a square matrix A, we will effectively perform QR factorization
* but if A is symmetric QR will be tridiagonal
*/
tuple<vector<vector<double> >, vector<vector<double> > > hessenberg(vector<vector<double> > A){
    // get the dims of the matrix
    int rows=A.size();
    int cols=A[0].size();
    // make V to store vj
    vector<vector<double> > V(rows, vector<double> (cols,0));

    // loop over columns of A
    for (int j = 0; j < cols-2; j++){
        // make vector s and v
        vector<double> s(rows);
        vector<double> v(rows,0);

        // compute entries of s
        double sum = 0.0;
        //iterate over the columns of A
        for (int i = j+1; i < rows; i++){
            sum = sum + pow(A[i][j],2);
        }
        // this trick: (x > 0) - (x < 0) yeilds +1, -1, 0 if sign(x)=+,-,0
        s[j] = ((A[j+1][j] > 0) - (A[j+1][j] < 0)) * sqrt(sum);
        // set the values of vj
        v[j+1] = A[j+1][j] + s[j];
        for (int i = j+2; i < rows; i++){
            v[i] = A[i][j];
        }
        
        // get the norm of vj
        double vnorm = twonorm(v);

        // rescale vj
        for (int i = 0; i < rows; i++){
            if (ismachzero(v[i]/vnorm)){
                v[i]=0.0;
            }
            else{
                v[i]=v[i]/vnorm;
            }   
        }
        // record vj in V
        for(int i=0; i<rows; i++){
            V[i][j]=v[i];
        }
        // === UPDATE A FROM THE LEFT ===
        // compute vv^T
        vector<vector<double> > vvt=outerproduct(v);

        // compute vvT A
        vector<vector<double> > vvtA=matrixmult(vvt,A);

        // update: A = A - 2HA
        for (int i = 0; i < rows; i++){
            for (int j2 = 0; j2 < cols; j2++){
                if (ismachzero(A[i][j2] - 2*vvtA[i][j2])){
                    A[i][j2] = 0.0;
                }
                else{
                    A[i][j2]= A[i][j2] - 2*vvtA[i][j2];    
                }
            }
        }

        // === UPDATE A FROM THE RIGHT ===
        // compute A vvT 
        vector<vector<double> > Avvt=matrixmult(A,vvt);
        // update: A = A - 2AH
        for (int i = 0; i < rows; i++){
            for (int j2 = 0; j2 < cols; j2++){
                if (ismachzero(A[i][j2] - 2*Avvt[i][j2])){
                    A[i][j2] = 0.0;
                }
                else{
                    A[i][j2]= A[i][j2] - 2*Avvt[i][j2];
                }
            }
        }

    }
    return {A, V};
}



// get difference in diagonal entries of A and B
double diagdiff(vector<vector<double> > A, vector<vector<double> > B){
    double sum = 0.0;
    for (unsigned long i = 0; i < A.size(); ++i){
        sum += (A[i][i]-B[i][i]);
    }
    return fabs(sum);
}

/*
* Eigenvalue QR w/out shift
* Given a matrix A, we iteratively calculate the QR factorization
* but on every iteration we replace A with RQ.
* We denote the condition of the iteration to be the significance of 
* of the change of the norm of the diagonals; if it is machine zero we consider 
* the algorithm to have converged to the diagonal matrix D_lambda.
*/

vector<vector<double> > iterQR(vector<vector<double> > A){
    // get sizes of A
    int rows = A.size();

    // make vector to record diagonals
    vector<double> delta(rows,0);
    for (int i = 0; i < rows; ++i){
        delta[i] = A[i][i];
    }
    // get first QR
    tuple<vector<vector<double> >, vector<vector<double> > > VRtuple = HHVR(A);
    vector<vector<double> > R = get<0>(VRtuple);

    // get Q from V
    vector<vector<double> > Q = QfromV(get<1>(VRtuple));

    // make new A
    A = matrixmult(R,Q);

    // update delta
    for (int i = 0; i < rows; ++i){
        delta[i] -= fabs(A[i][i]);
    }
    // now repeat until we don't record any changes
    while(!ismachzero(twonorm(delta))){
        // update delta
        for (int i = 0; i < rows; ++i){
            delta[i] = A[i][i];
        }
        // get next VR
        VRtuple = HHVR(A);
        R = get<0>(VRtuple);
        // get next Q
        Q = QfromV(get<1>(VRtuple));
        // make new A
        A = matrixmult(R,Q);
        // update delta
        for (int i = 0; i < rows; ++i){
            delta[i] -= A[i][i];
        }
    }
    // when we exit the while loop, we've found D_lambda
    return A;
}






/*
* Eigenvalue QR WITH shift
* Given a matrix A, we iteratively calculate the QR factorization of A - mu I
* but on every iteration we replace A with RQ + mu I.
* We denote the condition of the iteration to be the significance of 
* of the change of the norm of the diagonals; if it is machine zero we consider 
* the algorithm to have converged to the diagonal matrix D_lambda.
*/

vector<vector<double> > iterQRshifted(vector<vector<double> > A, double shift){
    // get sizes of A
    int rows = A.size();

    // do A-mu I
    for (int i = 0; i < rows; ++i){
        A[i][i] -= shift;
    }

    // make vector to record diagonals
    vector<double> delta(rows,0);
    for (int i = 0; i < rows; ++i){
        delta[i] = A[i][i];
    }
    // get first QR
    tuple<vector<vector<double> >, vector<vector<double> > > VRtuple = HHVR(A);
    vector<vector<double> > R = get<0>(VRtuple);

    // get Q from V
    vector<vector<double> > Q = QfromV(get<1>(VRtuple));

    // make new A
    A = matrixmult(R,Q);
    // do RQ + mu I
    for (int i = 0; i < rows; ++i){
        A[i][i] += shift;
    }

    // update delta
    for (int i = 0; i < rows; ++i){
        delta[i] -= fabs(A[i][i]);
    }
    // now repeat until we don't record any changes
    while(!ismachzero(twonorm(delta))){
        // update delta
        for (int i = 0; i < rows; ++i){
            delta[i] = A[i][i];
        }
        // do A-mu I
        for (int i = 0; i < rows; ++i){
            A[i][i] -= shift;
        }
        // get next VR
        VRtuple = HHVR(A);
        R = get<0>(VRtuple);
        // get next Q
        Q = QfromV(get<1>(VRtuple));
        // make new A
        A = matrixmult(R,Q);
        // do RQ + mu I
        for (int i = 0; i < rows; ++i){
            A[i][i] += shift;
        }
        // update delta
        for (int i = 0; i < rows; ++i){
            delta[i] -= A[i][i];
        }
    }
    // when we exit the while loop, we've found D_lambda
    return A;
}


/*
* Inverse Iteration
* Given a matrix A and a shift s
* iteratively approximate an eigenvector y until the change in y is insignificant
* returns a matrix of eigenvectors
*/
vector<vector<double> > inviter(vector<vector<double> > A, double shift){
    // get size
    int rows = A.size();
    // make delta to record change
    double delta = 0.0;

    // build initial guess of the eigenvector --> like this for my GE function
    vector<vector<double> > X(rows,vector<double>(1,1));

    // get A-shift I
    for (int i = 0; i < rows; ++i){
        A[i][i] -= shift;
    }

    // build y
    vector<vector<double> > Y(rows,vector<double>(1,1));

    // LU factor A
    // A= cholesky(A);
    // gaussian elimination
    tuple<vector<vector<double> >, vector<vector<double> > > out = GE(A, X);

    // solve for y
    // Y = choleskybacksub(A, X);
    Y = backsub(get<0>(out), get<1>(out));
    // normalize Y
    double norm = 0.0;
    for (int i = 0; i < rows; ++i){
        norm += Y[i][0]*Y[i][0];
    }
    norm = sqrt(norm);
    for (int i = 0; i < rows; ++i){
        Y[i][0]/=norm;
    }

    // record change in Y - X
    for (int i = 0; i < rows; ++i){
        delta += fabs(Y[i][0]-X[i][0]);
    }
    X = Y;

    // do while change > threshold
    while(!ismachzero(delta)){
        // update delta
        delta = 0.0;
        for (int i = 0; i < rows; ++i){
            delta += fabs(Y[i][0]-X[i][0]);
        }
        // solve for y
        // Y = choleskybacksub(A, X);
        out = GE(A, X);
        Y = backsub(get<0>(out), get<1>(out));
        // normalize Y
        norm = 0.0;
        for (int i = 0; i < rows; ++i){
            norm += Y[i][0]*Y[i][0];
        }
        norm = sqrt(norm);
        for (int i = 0; i < rows; ++i){
            Y[i][0]/=norm;
        }

        // record change in Y - X
        for (int i = 0; i < rows; ++i){
            delta -= fabs(Y[i][0]-X[i][0]);
        }
        X = Y;
    }
    // when we exit the while loop, we've found the eigenvector
    return X;
}




























