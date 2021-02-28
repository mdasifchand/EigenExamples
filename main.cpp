//
// Created by light on 01.03.21.
//

#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;
using namespace std;

bool COMPARE (Eigen::MatrixXf& A, Eigen::MatrixXf& B){

    if (A.norm() - B.norm() < 1e-4){

        return true;
    }

}



int main(){

/*  A = [ 1 2 3 4 ; 5 6 7 8 ; 9 10 11 12; 13 14 15 16]

 For simplicity reasons we would be considering this matrix. b = [ 1  1 1 1], in case we have to Ax=b

  */

// creating a Matrix

Matrix <double, 4,4, RowMajor> Matrix << 1, 2, 3, 4,  5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15, 16;

//sometimes or more than often people use typdef to simply the naming

typedef Matrix <double, 4,4, RowMajor> Matd44;

Matd44 Matrix2 << 1, 2, 3, 4,  5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15, 16;

bool value = COMPARE(Matrix, Matrix2);



}
