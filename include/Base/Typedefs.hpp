/**
 * @file Typedefs.hpp
 * @brief Some general typedefs used in the whole project 
 * @author Philippe Marti \<philippe.marti@colorado.edu\>
 */

#ifndef TYPEDEFS_HPP
#define TYPEDEFS_HPP

/// Generate maximum precision diagnositics output
#undef EIGEN_DEFAULT_IO_FORMAT
#define EIGEN_DEFAULT_IO_FORMAT IOFormat(Eigen::FullPrecision)

// Configuration includes
//
#include "SmartPointers/SharedPtrMacro.h"

// System includes
//
#include <complex>

// External includes
//
#include <Eigen/Core>
#include <Eigen/Sparse>

// Project includes
//
#include "Base/DecoupledComplex.hpp"

namespace QuICC {

   /**
    * @name Basic scalar types typedefs
    */
   //@{
   /// Typedef for floating point type value
   typedef double MHDFloat;
   /// Typedef for complex type value
   typedef std::complex<MHDFloat> MHDComplex;
   //@}

   /**
    * @name Array types typedefs
    */
   //@{
   /// Typedef for an array of bool values
   typedef Eigen::Matrix<bool, Eigen::Dynamic, 1>   ArrayB;
   /// Typedef for an array of integer values
   typedef Eigen::Matrix<int, Eigen::Dynamic, 1>   ArrayI;
   /// Typedef for an array of float values
   typedef Eigen::Matrix<MHDFloat, Eigen::Dynamic, 1>   Array;
   /// Typedef for an array of complex values
   typedef Eigen::Matrix<MHDComplex, Eigen::Dynamic, 1>   ArrayZ;
   //@}

   /**
    * @name Coefficient wise array types typedefs
    */
   //@{
   /// Typedef for an array of integer values
   typedef Eigen::Array<int, Eigen::Dynamic, 1>   ACoeffI;
   /// Typedef for an array of float values
   typedef Eigen::Array<MHDFloat, Eigen::Dynamic, 1>   ACoeff;
   /// Typedef for an array of complex values
   typedef Eigen::Array<MHDComplex, Eigen::Dynamic, 1>   ACoeffZ;
   //@}

   /**
    * @name Matrix types typedefs
    */
   //@{
   /// Typedef for an matrix of boolean values
   typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>   MatrixB;
   /// Typedef for a matrix of int values
   typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>   MatrixI;
   /// Typedef for a matrix of float values
   typedef Eigen::Matrix<MHDFloat, Eigen::Dynamic, Eigen::Dynamic>   Matrix;
   /// Typedef for a matrix of complex values
   typedef Eigen::Matrix<MHDComplex, Eigen::Dynamic, Eigen::Dynamic>   MatrixZ;
   /// Typedef for a decoupled real/imaginary complex matrix
   typedef Datatypes::DecoupledComplex<Matrix> DecoupledZMatrix;
   //@}

   /**
    * @name Sparse Matrix types typedefs
    */
   //@{
   /// Typedef for a sparse matrix of float values
   typedef Eigen::SparseMatrix<MHDFloat>   SparseMatrix;
   /// Typedef for a sparse matrix of complex values
   typedef Eigen::SparseMatrix<MHDComplex>   SparseMatrixZ;
   /// Typedef for a decoupled real/imaginary complex sparse matrix
   typedef Datatypes::DecoupledComplex<SparseMatrix> DecoupledZSparse;
   /// Typedef for the real triplets used to initialise sparse real matrices
   typedef Eigen::Triplet<MHDFloat>   Triplet;
   /// Typedef for the complex triplets used to initialise sparse complex matrices
   typedef Eigen::Triplet<MHDComplex>   TripletZ;
   //@}

   /**
    * @name Shared pointer typedefs
    */
   //@{
   /// Typedef for a smart reference counting pointer of an array of integers
   typedef SharedPtrMacro<ArrayI>   SharedArrayI;
   /// Typedef for an smart reference counting pointer of an array of reals
   typedef SharedPtrMacro<Array>   SharedArray;
   /// Typedef for an smart reference counting pointer of a matrix of reals
   typedef SharedPtrMacro<Matrix>   SharedMatrix;
   //@}
}

#endif // TYPEDEFS_HPP
