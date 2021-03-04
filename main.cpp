//
// Created by light on 01.03.21.
//

#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;
using namespace std;

bool COMPARE (Eigen::MatrixXd& A, Eigen::MatrixXd& B){

    if (A.norm() - B.norm() < 1e-4){

        return true;
    }

}



int main(){

/*  A = [ 1 2 3 4 ; 5 6 7 8 ; 9 10 11 12; 13 14 15 16]

 For simplicity reasons we would be considering this matrix. b = [ 1  1 1 1], in case we have to Ax=b

  */

// creating a Matrix

Eigen::Matrix <double, 4,4, Eigen::RowMajor> Matrix1;
Matrix1 << 1, 2, 3, 4,  5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15, 16;

// creating a Matrix 2nd way using a temporary matrix
Eigen::Matrix<double, 4,4, Eigen::RowMajor> Matrix2 = (Eigen::Matrix<double,4,4,RowMajor> () << 1, 2, 3, 4,  5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15, 16 ).finished();

std::cout << "The first matrix is  \n " << Matrix1  << " \n"
          << " Second matrix is    \n " << Matrix2 << std::endl;


/* create a row vector or column vector

 a. Vector2f, Vector3f, Vector4f This should give nx1 matrix (n rows and 1 column) something like [1 2 3 4]'
 b. RowVector2f, RowVector3f, RowVector4f Gives out  1xn (1 row and n columns), something like [1 2 3 4]
 c. VectorXf or VectorXd -> Dynamic Columns of floats or doubles.
 In general d implies double and f suffix implies it's float

*/

Vector2f V2 = (Vector2f() << 1,2).finished();
Vector3d V3;
V3 << 1,2,3 ;

RowVector2f  RV2 = (RowVector2f()<< 1,2 ).finished();

/*std::cout << "Default is a colum vector \n" <<
             " V2 is = \n" << V2 << "\n" <<
             " v3 is = \n " << V3 << std::endl;
*/
//std::cout << "RowVector2f is given by \n" << RV2 << std::endl;
// Other nicer ways to initialize a matrix are
Eigen::Matrix < double, 4,4, RowMajor > A_zero = Eigen::MatrixXd::Zero(4,4);
Matrix4d A_identity = MatrixXd::Identity(4,4);
Matrix4d A_setone = MatrixXd::Ones(4,4); // the r value is basically eye(4,4)

//reinitialize an existing Matrix
Matrix1.setZero();
//std::cout << "A_zero is \n" << A_zero << "\n" << "Matrix1 is reset to zero \n" << Matrix1 << std::endl;
Matrix1.setOnes();
Matrix1.setRandom();
//std::cout << "Matrix1 is reset to Random \n" << Matrix1 << std::endl;
Matrix1.setIdentity(); // Matrix1 = eye(N);

// Lengths of different tensors.

Matrix1.size(); // total size 4x4 = 16
Matrix1.rows(); //number of rows
Matrix1.cols(); // number of cols

// Element wise operations

//to access specific element
Matrix1(1,2);  // This should give out element at 2nd row and third cols, Remember the indices are i-1 and j-1 if you think in terms of matlab
// for vectors it's simply V2(1) or V2(2) etc.,

// to access specific row
Matrix2.row(0); // This gives out first row
std::cout << Matrix2.row(0) << std::endl;
Matrix2.col(0); // This gives out 1 col
std::cout << Matrix2.col(0) << std::endl;

// to change specific row or cols

Matrix2.col(0) << 1,2,3,4;
std::cout << Matrix2.col(0) << std::endl;
std::cout << Matrix2 << std::endl;

// to access a specific block;

Matrix2.block<2,2>(1, 1)  << 1.2, 1.3, 1.4, 1.5 ; // This is an interesting operation, it extracts the specific block and assigns value to it
// for example here we would like to extract a block of 2 rows and 2 columns as descrbed inside <> . And the numbers inside braces indicate the starting col and starting row
// here it's 1,1 which means 2,2 as the value is taken 1 less than actual value of rol or col
//std::cout << Matrix2 << std::endl;



// Resizing an existing matrix, matrix1 is 4x4. This can resized to 2x8, 8x2, 1x16 and 16x1
//Matrix1.resize(2,8); // This normally fails due to assertations, i wouldn't wanna tamper with assertations at this stage
//std::cout << Matrix1  << std::endl;


// Stacking Vectors or Matrices
Matrix1.setRandom();
MatrixXd M( Matrix1.rows() + Matrix1.rows()+ Matrix1.rows(), Matrix1.cols()); // I could have written Matrix1.rows()*3, in case there were m << A,B,C; then we need to write em individually
M << Matrix1, Matrix1, Matrix1;
std::cout << M << std::endl;


// Filling all the elements with some constant value

Matrix2.fill(1.0);
std::cout << Matrix2 << std::endl;


VectorXd VX = VectorXd::LinSpaced(4,1,5) ;       // linspace(low,high,size)'
VX.setLinSpaced(4,1,5);               // v = linspace(low,high,size)'
std::cout << VX << std::endl;

// Matrix slicing
// blocks -> always go for templated version, it has better speed up
// there are two types of blocks one with vectors and second one with Matrix

// for vectors, you simply pass on a single value
int const n = 2;
VX.head<n>() << 1, 2; // for head initial 2 values
std::cout << VX << std::endl;
VX.tail<n>() << 3,4; // bottom two values or N-n till N

// for matrix there is a block actually, you essentially pass on initial inidices of starting with size of the matrix as shown in line 101
// Other try outs
/*
    P.col(j)                           // P(:, j+1)
    P.leftCols<cols>()                 // P(:, 1:cols)
    P.leftCols(cols)                   // P(:, 1:cols)
    P.middleCols<cols>(j)              // P(:, j+1:j+cols)
    P.middleCols(j, cols)              // P(:, j+1:j+cols)
    P.rightCols<cols>()                // P(:, end-cols+1:end)
    P.rightCols(cols)                  // P(:, end-cols+1:end)
    P.topRows<rows>()                  // P(1:rows, :)
    P.topRows(rows)                    // P(1:rows, :)
    P.middleRows<rows>(i)              // P(i+1:i+rows, :)
    P.middleRows(i, rows)              // P(i+1:i+rows, :)
    P.bottomRows<rows>()               // P(end-rows+1:end, :)
    P.bottomRows(rows)                 // P(end-rows+1:end, :)
    P.topLeftCorner(rows, cols)        // P(1:rows, 1:cols)
    P.topRightCorner(rows, cols)       // P(1:rows, end-cols+1:end)
    P.bottomLeftCorner(rows, cols)     // P(end-rows+1:end, 1:cols)
    P.bottomRightCorner(rows, cols)    // P(end-rows+1:end, end-cols+1:end)
    P.topLeftCorner<rows,cols>()       // P(1:rows, 1:cols)
    P.topRightCorner<rows,cols>()      // P(1:rows, end-cols+1:end)
    P.bottomLeftCorner<rows,cols>()    // P(end-rows+1:end, 1:cols)
    P.bottomRightCorner<rows,cols>()   // P(end-rows+1:end, end-cols+1:end)
*/

// Views, transpose, etc;
// Eigen                           // Matlab
 /*    R.adjoint()                        // R'
    R.transpose()                      // R.' or conj(R')       // Read-write
    R.diagonal()                       // diag(R)               // Read-write
    x.asDiagonal()                     // diag(x)
    R.transpose().colwise().reverse()  // rot90(R)              // Read-write
    R.rowwise().reverse()              // fliplr(R)
    R.colwise().reverse()              // flipud(R)
    R.replicate(i,j)                   // repmat(P,i,j)
*/


    // All the same as Matlab, but matlab doesn't have *= style operators.
// Matrix-vector.  Matrix-matrix.   Matrix-scalar.
  /*
    y  = M*x;          R  = P*Q;        R  = P*s;
    a  = b*M;          R  = P - Q;      R  = s*P;
    a *= M;            R  = P + Q;      R  = P/s;
    R *= Q;          R  = s*P;
    R += Q;          R *= s;
    R -= Q;          R /= s;

   // Vectorized operations on each element independently
// Eigen                       // Matlab
R = P.cwiseProduct(Q);         // R = P .* Q
R = P.array() * s.array();     // R = P .* s
R = P.cwiseQuotient(Q);        // R = P ./ Q
R = P.array() / Q.array();     // R = P ./ Q
R = P.array() + s.array();     // R = P + s
R = P.array() - s.array();     // R = P - s
R.array() += s;                // R = R + s
R.array() -= s;                // R = R - s
R.array() < Q.array();         // R < Q
R.array() <= Q.array();        // R <= Q
R.cwiseInverse();              // 1 ./ P
R.array().inverse();           // 1 ./ P
R.array().sin()                // sin(P)
R.array().cos()                // cos(P)
R.array().pow(s)               // P .^ s
R.array().square()             // P .^ 2
R.array().cube()               // P .^ 3
R.cwiseSqrt()                  // sqrt(P)
R.array().sqrt()               // sqrt(P)
R.array().exp()                // exp(P)
R.array().log()                // log(P)
R.cwiseMax(P)                  // max(R, P)
R.array().max(P.array())       // max(R, P)
R.cwiseMin(P)                  // min(R, P)
R.array().min(P.array())       // min(R, P)
R.cwiseAbs()                   // abs(P)
R.array().abs()                // abs(P)
R.cwiseAbs2()                  // abs(P.^2)
R.array().abs2()               // abs(P.^2)
(R.array() < s).select(P,Q );  // (R < s ? P : Q)
R = (Q.array()==0).select(P,A) // R(Q==0) = P(Q==0)
R = P.unaryExpr(ptr_fun(func)) // R = arrayfun(func, P)   // with: scalar func(const scalar &x

//// Type conversion
// Eigen                  // Matlab
A.cast<double>();         // double(A)
A.cast<float>();          // single(A)
A.cast<int>();            // int32(A)
A.real();                 // real(A)
A.imag();                 // imag(A)
// if the original type equals destination type, no work is done

// Note that for most operations Eigen requires all operands to have the same type:
MatrixXf F = MatrixXf::Zero(3,3);
A += F;                // illegal in Eigen. In Matlab A = A+F is allowed
A += F.cast<double>(); // F converted to double and then added (generally, conversion happens on-the-fly)

// Eigen can map existing memory into Eigen matrices.
float array[3];
Vector3f::Map(array).fill(10);            // create a temporary Map over array and sets entries to 10
int data[4] = {1, 2, 3, 4};
Matrix2i mat2x2(data);                    // copies data into mat2x2
Matrix2i::Map(data) = 2*mat2x2;           // overwrite elements of data with 2*mat2x2
MatrixXi::Map(data, 2, 2) += mat2x2;      // adds mat2x2 to elements of data (alternative syntax if size is not know at compile time)

// Solve Ax = b. Result stored in x. Matlab: x = A \ b.
x = A.ldlt().solve(b));  // A sym. p.s.d.    #include <Eigen/Cholesky>
x = A.llt() .solve(b));  // A sym. p.d.      #include <Eigen/Cholesky>
x = A.lu()  .solve(b));  // Stable and fast. #include <Eigen/LU>
x = A.qr()  .solve(b));  // No pivoting.     #include <Eigen/QR>
x = A.svd() .solve(b));  // Stable, slowest. #include <Eigen/SVD>
// .ldlt() -> .matrixL() and .matrixD()
// .llt()  -> .matrixL()
// .lu()   -> .matrixL() and .matrixU()
// .qr()   -> .matrixQ() and .matrixR()
// .svd()  -> .matrixU(), .singularValues(), and .matrixV()

// Eigenvalue problems
// Eigen                          // Matlab
A.eigenvalues();                  // eig(A);
EigenSolver<Matrix3d> eig(A);     // [vec val] = eig(A)
eig.eigenvalues();                // diag(val)
eig.eigenvectors();               // vec
// For self-adjoint matrices use SelfAdjointEigenSolver<>

This was copied from

*/

}
