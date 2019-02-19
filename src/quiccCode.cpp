#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/LU>
#include <iostream>
#include "PolynomialTransforms/AssociatedLegendrePolynomial.hpp"

//Associated legendre polynomial
void plm( Eigen::Ref<QuICC::Matrix> op, int m, QuICC::Array igrid){
    QuICC::Matrix mOp(op);
    QuICC::Matrix ipoly(op);
    QuICC::Polynomial::AssociatedLegendrePolynomial::Plm(mOp, ipoly, m, igrid);
    op = mOp;
}

int main(){
return 0;
}
