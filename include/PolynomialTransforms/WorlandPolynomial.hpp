/** 
 * @file WorlandPolynomial.hpp
 * @brief Implementation of the Worland polynomial
 * @author Philippe Marti \<philippe.marti@colorado.edu\>
 */

#ifndef WORLANDPOLYNOMIAL_HPP
#define WORLANDPOLYNOMIAL_HPP

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
#include "PolynomialTransforms/ThreeTermRecurrence.hpp"

namespace QuICC {

namespace Polynomial {

   /**
    * @brief Implementation of the Worland polynomial
    */ 
   class WorlandPolynomial
   {
      public:
         /**
          * @brief Compute \f$W_n^l (r)\f$
          *
          * Internal computation can be done in multiple precision
          */
         static void Wnl(Matrix& poly, internal::Matrix& ipoly, const int l, const internal::Array& igrid);

         /**
          * @brief Compute \f$\frac{W_n^l (r)}{r}\f$
          *
          * Internal computation can be done in multiple precision
          */
         static void r_1Wnl(Matrix& poly, internal::Matrix& iwnl, const int l, const internal::Array& igrid);

         /**
          * @brief Compute \f$\frac{d}{dr} W_n^l (r)\f$
          *
          * Internal computation can be done in multiple precision
          */
         static void dWnl(Matrix& diff, internal::Matrix& idiff, const int l, const internal::Array& igrid);

         /**
          * @brief Compute \f$\frac{d}{dr} r W_n^l (r)\f$
          *
          * Internal computation can be done in multiple precision
          */
         static void drWnl(Matrix& diff, internal::Matrix& idiff, const int l, const internal::Array& igrid);

         /**
          * @brief Compute \f$\frac{1}{r}\frac{d}{dr} r W_n^l (r)\f$
          *
          * Internal computation can be done in multiple precision
          */
         static void r_1drWnl(Matrix& diff, internal::Matrix& idiff, const int l, const internal::Array& igrid);

         /**
          * @brief Compute \f$\frac{d}{dr}\frac{1}{r}\frac{d}{dr} r W_n^l (r)\f$
          *
          * Internal computation can be done in multiple precision
          */
         static void dr_1drWnl(Matrix& diff, internal::Matrix& idiff, const int l, const internal::Array& igrid);

         /**
          * @brief Compute spherical \f$\nabla^2 W_n^l (r)\f$
          *
          * Internal computation can be done in multiple precision
          */
         static void slaplWnl(Matrix& poly, internal::Matrix& iwnl, const int l, const internal::Array& igrid);

         /**
          * @brief Compute cylindrical horizontal \f$\nabla_h^2 W_n^l (r)\f$
          *
          * Internal computation can be done in multiple precision
          */
         static void claplhWnl(Matrix& poly, internal::Matrix& iwnl, const int l, const internal::Array& igrid);

         /**
          * @brief Compute cylindrical horizontal \f$\frac{1}{r}\nabla_h^2 W_n^l (r)\f$
          *
          * Internal computation can be done in multiple precision
          */
         static void r_1claplhWnl(Matrix& poly, internal::Matrix& iwnl, const int l, const internal::Array& igrid);

         /**
          * @brief Compute cylindrical horizontal \f$\frac{d}{dr}\nabla_h^2 W_n^l (r)\f$
          *
          * Internal computation can be done in multiple precision
          */
         static void dclaplhWnl(Matrix& poly, internal::Matrix& iwnl, const int l, const internal::Array& igrid);

         /**
          * @brief Compute \f$W_0^l (r)\f$
          *
          * Internal computation can be done in multiple precision
          */
         static void W0l(Eigen::Ref<internal::Matrix> iw0ab, const int l, const internal::MHDFloat alpha, const internal::MHDFloat beta, const internal::Array& igrid, ThreeTermRecurrence::NormalizerAB norm);

         /**
          * @brief Compute \f$\frac{d}{dr} W_n^0 (r)\f$
          *
          * Internal computation can be done in multiple precision
          */
         static void dWn0(Matrix& diff, internal::Matrix& idiff, const internal::Array& igrid);

         static ThreeTermRecurrence::NormalizerNAB normWPnab();
         static ThreeTermRecurrence::NormalizerAB normWP1ab();
         static ThreeTermRecurrence::NormalizerAB normWP0ab();
         static ThreeTermRecurrence::NormalizerNAB normWDPnab();
         static ThreeTermRecurrence::NormalizerAB normWDP1ab();
         static ThreeTermRecurrence::NormalizerAB normWDP0ab();
         static ThreeTermRecurrence::NormalizerNAB normWD2Pnab();
         static ThreeTermRecurrence::NormalizerAB normWD2P1ab();
         static ThreeTermRecurrence::NormalizerAB normWD2P0ab();
         static ThreeTermRecurrence::NormalizerNAB normWD3Pnab();
         static ThreeTermRecurrence::NormalizerAB normWD3P1ab();
         static ThreeTermRecurrence::NormalizerAB normWD3P0ab();

         /**
          * @brief Polynomial normalizer for unit Worland normalization
          */
         static internal::Array unitWPnab(const internal::MHDFloat dn, const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief Polynomial n=0 normalizer for unit Worland normalization
          */
         static internal::Array unitWP0ab(const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief Polynomial n=1 normalizer for unit Worland normalization
          */
         static internal::Array unitWP1ab(const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief First derivative normalizer for unit Worland normalization
          */
         static internal::Array unitWDPnab(const internal::MHDFloat dn, const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief First derivative n=0 normalizer for unit Worland normalization
          */
         static internal::Array unitWDP0ab(const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief First derivative n=1 normalizer for unit Worland normalization
          */
         static internal::Array unitWDP1ab(const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief Second derivative normalizer for unit Worland normalization
          */
         static internal::Array unitWD2Pnab(const internal::MHDFloat dn, const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief Second derivative n=0 normalizer for unit Worland normalization
          */
         static internal::Array unitWD2P0ab(const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief Second derivative n=1 normalizer for unit Worland normalization
          */
         static internal::Array unitWD2P1ab(const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief Third derivative normalizer for unit Worland normalization
          */
         static internal::Array unitWD3Pnab(const internal::MHDFloat dn, const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief Third derivative n=0 normalizer for unit Worland normalization
          */
         static internal::Array unitWD3P0ab(const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief Third derivative n=1 normalizer for unit Worland normalization
          */
         static internal::Array unitWD3P1ab(const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief Polynomial normalizer for natural normalization
          */
         static internal::Array naturalWPnab(const internal::MHDFloat n, const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief Polynomial n=0 normalizer for natural normalization
          */
         static internal::Array naturalWP0ab(const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief Polynomial n=1 normalizer for natural normalization
          */
         static internal::Array naturalWP1ab(const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief First derivative normalizer for natural normalization
          */
         static internal::Array naturalWDPnab(const internal::MHDFloat n, const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief First derivative n=0 normalizer for natural normalization
          */
         static internal::Array naturalWDP0ab(const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief First derivative n=1 normalizer for natural normalization
          */
         static internal::Array naturalWDP1ab(const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief Second derivative normalizer for natural normalization
          */
         static internal::Array naturalWD2Pnab(const internal::MHDFloat n, const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief Second derivative n=0 normalizer for natural normalization
          */
         static internal::Array naturalWD2P0ab(const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief Second derivative n=1 normalizer for natural normalization
          */
         static internal::Array naturalWD2P1ab(const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief Third derivative normalizer for natural normalization
          */
         static internal::Array naturalWD3Pnab(const internal::MHDFloat n, const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief Third derivative n=0 normalizer for natural normalization
          */
         static internal::Array naturalWD3P0ab(const internal::MHDFloat a, const internal::MHDFloat b);

         /**
          * @brief Third derivative n=1 normalizer for natural normalization
          */
         static internal::Array naturalWD3P1ab(const internal::MHDFloat a, const internal::MHDFloat b);


      private:
         /**
          * @brief Get alpha parameter of Jacobi polynomial
          */
         static internal::MHDFloat alpha(const int l);

         /**
          * @brief Get beta parameter of Jacobi polynomial
          */
         static internal::MHDFloat beta(const int l);

         /**
          * @brief Constructor
          */
         WorlandPolynomial();

         /**
          * @brief Destructor
          */
         ~WorlandPolynomial();

   };
}
}

#endif // WORLANDPOLYNOMIAL_HPP
