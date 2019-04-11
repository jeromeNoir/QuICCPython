/**
 * @file DecoupledComplex.hpp
 * @brief Store complex numbers as two independent matrices
 * @author Philippe Marti \<philippe.marti@colorado.edu\>
 */

#ifndef DECOUPLEDCOMPLEX_HPP
#define DECOUPLEDCOMPLEX_HPP

// System includes
//

// External includes
//

// Project includes
//

namespace QuICC {

namespace Datatypes  {

   /**
    * @brief Store complex numbers as two independent matrices
    */
   template <typename TData> class DecoupledComplex
   {
      public:
         /// Typedef for the Scalar type (This is NOT the internal scalar type, it is for compatibility with Eigen's matrices)
         typedef std::complex<double> Scalar;

         /**
          * @brief Constructor for empty matrices
          */
         DecoupledComplex();

         /**
          * @brief Constructor with identical real and imaginary sizes
          */
         DecoupledComplex(const int rows, const int cols);

         /**
          * @brief Constructor with different real and imaginary sizes
          */
         DecoupledComplex(const int rowsR, const int colsR, const int rowsI, const int colsI);

         /**
          * @brief Empty Destructor
          */
         ~DecoupledComplex();

         /**
          * @brief Get real component
          */
         const TData& real() const;

         /**
          * @brief Get imaginary component
          */
         const TData& imag() const;

         /**
          * @brief Set real component
          */
         TData& real();

         /**
          * @brief Set imaginary component
          */
         TData& imag();

         /**
          * @brief Set both components to zero
          */
         void setZero();
         
      protected:

      private:
         /**
          * @brief Real component
          */
         TData mReal;

         /**
          * @brief Imaginary component
          */
         TData mImag;
   };

   template <typename TData> inline const TData& DecoupledComplex<TData>::real() const
   {
      return this->mReal;
   }

   template <typename TData> inline const TData& DecoupledComplex<TData>::imag() const
   {
      return this->mImag;
   }

   template <typename TData> inline TData& DecoupledComplex<TData>::real()
   {
      return this->mReal;
   }

   template <typename TData> inline TData& DecoupledComplex<TData>::imag()
   {
      return this->mImag;
   }

   template <typename TData> inline void DecoupledComplex<TData>::setZero()
   {
      this->mReal.setZero();
      this->mImag.setZero();
   }

   template <typename TData> DecoupledComplex<TData>::DecoupledComplex()
      : mReal(), mImag()
   {
   }

   template <typename TData> DecoupledComplex<TData>::DecoupledComplex(const int rows, const int cols)
      : mReal(rows,cols), mImag(rows,cols)
   {
   }

   template <typename TData> DecoupledComplex<TData>::DecoupledComplex(const int rowsR, const int colsR, const int rowsI, const int colsI)
      : mReal(rowsR,rowsR), mImag(rowsI,colsI)
   {
   }

   template <typename TData> DecoupledComplex<TData>::~DecoupledComplex()
   {
   }

}
}

#endif // DECOUPLEDCOMPLEX_HPP
