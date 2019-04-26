/**
 * @file Precision.hpp
 * @brief Small wrapper class for generic normal of multiple precision internal computations 
 * @author Philippe Marti \<philippe.marti@colorado.edu\>
 */

#ifndef PRECISION_HPP
#define PRECISION_HPP

// Configuration includes
//

// System includes
//

// External includes
//

// Project includes
//
#include "Base/Typedefs.hpp"

#ifdef QUICC_MULTPRECISION
   #include "Base/MpTypedefs.hpp"
   
   /// Define a small macro to replace float constants to strings in the case of MP computations
   #define MHD_MP(c) #c
#else
   #ifdef QUICC_SMARTPTR_CXX0X 
      #include <cmath>
   #else 
      #include <tr1/cmath>
   #endif 
   /// For normal computations the macro does nothing
   #define MHD_MP(c) c
#endif // QUICC_MULTPRECISION

namespace QuICC {

   namespace internal {

      #ifdef QUICC_MULTPRECISION
      //////////////////////////////////////////////////////////////////         
         /// Typedef for the internal float type
         typedef QuICC::MHDMpFloat MHDFloat;

         /// Typedef for the internal Array type
         typedef QuICC::MpArray Array;

         /// Typedef for the internal Matrix type
         typedef QuICC::MpMatrix Matrix;

         /// Typedef for the smart internal Array type
         typedef QuICC::SharedMpArray SharedArray;

         /// Typedef for the smart internal Matrix type
         typedef QuICC::SharedMpMatrix SharedMatrix;
      //////////////////////////////////////////////////////////////////
      #else
      //////////////////////////////////////////////////////////////////
         /// Typedef for the internal float type
         typedef QuICC::MHDFloat MHDFloat;

         /// Typedef for the internal Array type
         typedef QuICC::Array Array;

         /// Typedef for the internal Matrix type
         typedef QuICC::Matrix Matrix;

         /// Typedef for the smart internal Array type
         typedef QuICC::SharedArray SharedArray;

         /// Typedef for the smart internal Matrix type
         typedef QuICC::SharedMatrix SharedMatrix;
      //////////////////////////////////////////////////////////////////
      #endif // QUICC_MULTPRECISION
   }

   /**
    * @brief Simple class holding some typedefs to allow for internal MP computations
    */
   class Precision
   {
      public:

         /**
          * @brief Precision dependent mathematical constant \f$\pi\f$
          */
         static const internal::MHDFloat PI;

         /**
          * @brief Initialise the precision setup
          */
         static void init();

         /**
          * @brief Cast the internal smart Array to an external one
          *
          * @param spIArr Internal smart Array to cast
          */
         static SharedArray cast(internal::SharedArray spIArr);

         /**
          * @brief Cast the internal smart Matrix to an external one
          *
          * @param spIMat Internal smart Matrix to cast
          */
         static SharedMatrix cast(internal::SharedMatrix spIMat);

         /**
          * @brief Cast the internal smart Array to an external one
          *
          * @param rIArr Internal Array to cast
          */
         static Array cast(const internal::Array& rIArr);

         /**
          * @brief Cast the internal smart Matrix to an external one
          *
          * @param rIMat Internal Matrix to cast
          */
         static Matrix cast(const internal::Matrix& rIMat);

      private:
         /**
         * @brief Empty constructor
         */
         Precision();

         /**
         * @brief Simple empty destructor
         */
         ~Precision();
   };

   inline void Precision::init()
   {
      #ifdef QUICC_MULTPRECISION
         mpfr::mpreal::set_default_prec(256);
      #endif // QUICC_MULTPRECISION
   }

   inline SharedArray Precision::cast(internal::SharedArray spIArr)
   {
      #ifdef QUICC_MULTPRECISION
         SharedArray spArr(new Array(spIArr->size()));

         // Loop over whole array
         for(int i=0; i < spIArr->size(); i++)
         {
            (*spArr)(i) = (*spIArr)(i).toDouble();
         }

         return spArr;
      #else
         return spIArr;
      #endif // QUICC_MULTPRECISION
   }

   inline SharedMatrix Precision::cast(internal::SharedMatrix spIMat)
   {
      #ifdef QUICC_MULTPRECISION
         SharedMatrix spMat(new Matrix(spIMat->rows(),spIMat->cols()));

         // Loop over whole matrix
         for(int j=0; j < spIMat->cols(); j++)
         {
            for(int i=0; i < spIMat->rows(); i++)
            {
               (*spMat)(i,j) = (*spIMat)(i,j).toDouble();
            }
         }

         return spMat;
      #else
         return spIMat;
      #endif // QUICC_MULTPRECISION
   }

   inline Array Precision::cast(const internal::Array& rIArr)
   {
      #ifdef QUICC_MULTPRECISION
         Array arr(rIArr.size());

         // Loop over whole array
         for(int i=0; i < rIArr.size(); i++)
         {
            arr(i) = rIArr(i).toDouble();
         }

         return arr;
      #else
         return rIArr;
      #endif // QUICC_MULTPRECISION
   }

   inline Matrix Precision::cast(const internal::Matrix& rIMat)
   {
      #ifdef QUICC_MULTPRECISION
         Matrix mat(rIMat.rows(),rIMat.cols());

         // Loop over whole matrix
         for(int j=0; j < rIMat.cols(); j++)
         {
            for(int i=0; i < rIMat.rows(); i++)
            {
               mat(i,j) = rIMat(i,j).toDouble();
            }
         }

         return mat;
      #else
         return rIMat;
      #endif // QUICC_MULTPRECISION
   }

#ifdef QUICC_MULTPRECISION
   /// Create a namespace alias for the internal precision stuff pointing to mpfr namespace
   namespace  precision = mpfr;
   namespace  precisiontr1 = mpfr;
#else
   /// Create a namespace alias for the internal precision stuff pointing to std namespace
   namespace  precision = std;
   #ifdef QUICC_SMARTPTR_CXX0X 
      namespace  precisiontr1 = std;
   #else
      namespace  precisiontr1 = std::tr1;
   #endif 

#endif // QUICC_MULTPRECISION
}

#endif // PRECISION_HPP
