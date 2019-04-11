/** 
 * @file AssociatedLegendrePolynomial.hpp
 * @brief Implementation of the associated Legendre polynomial
 * @author Philippe Marti \<philippe.marti@colorado.edu\>
 */

#ifndef ASSOCIATEDLEGENDREPOLYNOMIAL_HPP
#define ASSOCIATEDLEGENDREPOLYNOMIAL_HPP

// Debug includes
//

// Configuration includes
//

// System includes
//

// External includes
//

// Project includes
//
#include "Base/Precision.hpp"

namespace QuICC {

namespace Polynomial {

   /**
    * @brief Implementation of the associated Legendre polynomial
    */ 
   class AssociatedLegendrePolynomial
   {
      public:
         /**
          * @brief Compute the associated Legendre \f$P_l^m (\cos\theta)\f$ for all l
          *
          * Internal computation can be done in multiple precision
          */
         static void Plm(Matrix& poly, internal::Matrix& ipoly, const int m, const internal::Array& igrid);

         /**
          * @brief Compute the associated Legendre \f$\frac{d}{d_\theta} P_l^m (\cos\theta)\f$ for all l (scheme A)
          *
          * Internal computation can be done in multiple precision
          */
         static void dPlmA(Matrix& diff, internal::Matrix& idiff, const int m, const internal::Matrix& ipoly, const internal::Array& igrid);

         /**
          * @brief Compute the associated Legendre \f$\frac{d}{d_\theta} P_l^m (\cos\theta)\f$ for all l (scheme B)
          *
          * Internal computation can be done in multiple precision
          */
         static void dPlmB(Matrix& diff, internal::Matrix& idiff, const int m, const internal::Array& igrid);

         /**
          * @brief Compute the associated Legendre \f$\frac{P_l^m (\cos\theta)}{\sin\theta}\f$ for all l
          *
          * Internal computation can be done in multiple precision
          */
         static void sin_1Plm(Matrix& poly, internal::Matrix& ipoly, const int m, const internal::Array& igrid);

         /**
          * @brief Compute the associated Legendre \f$P_l^m (\cos\theta)\f$
          *
          * Internal computation can be done in multiple precision
          */
         static void Plm(Eigen::Ref<internal::Matrix> iplm, const int m, const int l, const Eigen::Ref<const internal::Matrix>& ipl_1m, const Eigen::Ref<const internal::Matrix>& ipl_2m, const internal::Array& igrid);

         /**
          * @brief Compute the associated Legendre \f$\frac{d}{d_\theta} P_l^m (\cos\theta)\f$ (scheme A)
          *
          * Internal computation can be done in multiple precision. Uses the derivative of the recurrence relation.
          */
         static void dPlmA(Eigen::Ref<internal::Matrix> idplm, const int m, const int l, const Eigen::Ref<const internal::Matrix>& idpl_1m, const Eigen::Ref<const internal::Matrix>& idpl_2m, const Eigen::Ref<const internal::Matrix>& ipl_1m, const internal::Array& igrid);

         /**
          * @brief Compute the associated Legendre \f$\frac{d}{d_\theta} P_l^m (\cos\theta)\f$ (scheme B)
          *
          * Internal computation can be done in multiple precision. Uses recurrence relation for the wanted expression.
          */
         static void dPlmB(Eigen::Ref<internal::Matrix> idplm, const int m, const int l, const Eigen::Ref<const internal::Matrix>& iplm_1, const Eigen::Ref<const internal::Matrix>& iplm1);

         /**
          * @brief Compute the associated Legendre \f$\frac{d}{d_\theta} P_l (\cos\theta)\f$ (scheme B)
          *
          * Internal computation can be done in multiple precision. Uses recurrence relation for the wanted expression.
          */
         static void dPl0B(Eigen::Ref<internal::Matrix> idplm, const int l, const Eigen::Ref<const internal::Matrix>& iplm1);

         /**
          * @brief Compute the associated Legendre \f$\frac{P_l^m (\cos\theta)}{\sin\theta}\f$
          *
          * Internal computation can be done in multiple precision
          */
         static void sin_1Plm(Eigen::Ref<internal::Matrix> ipl1m, const int m, const int l, const Eigen::Ref<const internal::Matrix>& ipl1m1, const Eigen::Ref<const internal::Matrix>& ipl1m_1);

         /**
          * @brief Compute the associated Legendre \f$P_m^m (\cos\theta)\f$
          *
          * Internal computation can be done in multiple precision
          */
         static void Pmm(Eigen::Ref<internal::Matrix> op, const int m, const internal::Array& igrid);

         /**
          * @brief Compute the associated Legendre \f$P_{m+1}^m (\cos\theta)\f$
          *
          * Internal computation can be done in multiple precision
          */
         static void Pm1m(Eigen::Ref<internal::Matrix> op, const int m, const Eigen::Ref<const internal::Matrix>& ipmm, const internal::Array& igrid);

         /**
          * @brief Compute the associated Legendre \f$\frac{d}{d_\theta} P_m^m (\cos\theta)\f$ (scheme A)
          *
          * Internal computation can be done in multiple precision
          */
         static void dPmmA(Eigen::Ref<internal::Array> op, const int m, const internal::Array& igrid);

         /**
          * @brief Compute the associated Legendre \f$\frac{d}{d_\theta} P_m^m (\cos\theta)\f$ (scheme B)
          *
          * Internal computation can be done in multiple precision
          */
         static void dPmmB(Eigen::Ref<internal::Array> op, const int m, const Eigen::Ref<const internal::Array>& iplm_1);

         /**
          * @brief Compute the associated Legendre \f$\frac{d}{d_\theta} P_{m+1}^m (\cos\theta)\f$ (scheme A)
          *
          * Internal computation can be done in multiple precision
          */
         static void dPm1mA(Eigen::Ref<internal::Array> op, const int m, const Eigen::Ref<const internal::Array>& ipmm, const Eigen::Ref<const internal::Array>& idpmm, const internal::Array& igrid);

      private:
         /**
          * @brief Constructor
          */
         AssociatedLegendrePolynomial();

         /**
          * @brief Destructor
          */
         ~AssociatedLegendrePolynomial();

   };
}
}

#endif // ASSOCIATEDLEGENDREPOLYNOMIAL_HPP
