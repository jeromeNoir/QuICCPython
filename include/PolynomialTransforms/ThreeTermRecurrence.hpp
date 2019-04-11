/** 
 * @file ThreeTermRecurrence.hpp
 * @brief General implementation for three term recurrence relations
 * @author Philippe Marti \<philippe.marti@colorado.edu\>
 */

#ifndef THREETERMRECURRENCE_HPP
#define THREETERMRECURRENCE_HPP

// Debug includes
//

// Configuration includes
//

// System includes
//
#ifdef QUICC_SMARTPTR_CXX0X
   #include <functional>
   #define FuncMacro std
#else
   #include <tr1/functional>
   #define FuncMacro std::tr1
#endif

// External includes
//

// Project includes
//
#include "Base/Precision.hpp"

namespace QuICC {

namespace Polynomial {

   /**
    * @brief Implementation of the Jacobi polynomial
    */ 
   class ThreeTermRecurrence
   {
      public:
         /// Typedef for the function signature of an n independent constant normalizer 
         typedef FuncMacro::function<internal::Array()> NormalizerC;

         /// Typedef for the function signature of an n dependent constant normalizer 
         typedef FuncMacro::function<internal::Array(const internal::MHDFloat)> NormalizerNC;

         /// Typedef for the function signature of an n independent one parameter normalizer 
         typedef FuncMacro::function<internal::Array(const internal::MHDFloat)> NormalizerA;

         /// Typedef for the function signature of an n dependent one parameter normalizer 
         typedef FuncMacro::function<internal::Array(const internal::MHDFloat, const internal::MHDFloat)> NormalizerNA;

         /// Typedef for the function signature of an n independent two parameter normalizer 
         typedef FuncMacro::function<internal::Array(const internal::MHDFloat, const internal::MHDFloat)> NormalizerAB;

         /// Typedef for the function signature of an n dependent two parameter normalizer 
         typedef FuncMacro::function<internal::Array(const internal::MHDFloat, const internal::MHDFloat, const internal::MHDFloat)> NormalizerNAB;

         /**
          * @brief Compute three term recurrence for $P_n(x)$ normalizer
          *
          * Internal computation can be done in multiple precision
          */
         static void Pn(Eigen::Ref<internal::Matrix> ipn, const int n, const Eigen::Ref<const internal::Matrix>& ipn_1, const Eigen::Ref<const internal::Matrix>& ipn_2, const internal::Array& igrid, NormalizerNC norm);

         /**
          * @brief Compute three term recurrence for $P_1(x)$ normalizer
          *
          * Internal computation can be done in multiple precision
          */
         static void P1(Eigen::Ref<internal::Matrix> ip1, const Eigen::Ref<const internal::Matrix>& ip0, const internal::Array& igrid, NormalizerC norm);

         /**
          * @brief Compute three term recurrence for $P_0(x)$ normalizer
          *
          * Internal computation can be done in multiple precision
          */
         static void P0(Eigen::Ref<internal::Matrix> ip0, const internal::Array& igrid, NormalizerC norm);

         /**
          * @brief Compute three term recurrence for $P_n(x)^{(\alpha)}$ with one parameter normalizer
          *
          * Internal computation can be done in multiple precision
          */
         static void Pn(Eigen::Ref<internal::Matrix> ipn, const int n, const internal::MHDFloat alpha, const Eigen::Ref<const internal::Matrix>& ipn_1, const Eigen::Ref<const internal::Matrix>& ipn_2, const internal::Array& igrid, NormalizerNA norm);

         /**
          * @brief Compute three term recurrence for $P_1(x)^{(\alpha)}$ with one parameter normalizer
          *
          * Internal computation can be done in multiple precision
          */
         static void P1(Eigen::Ref<internal::Matrix> ip1, const internal::MHDFloat alpha, const Eigen::Ref<const internal::Matrix>& ip0, const internal::Array& igrid, NormalizerA norm);

         /**
          * @brief Compute three term recurrence for $P_0(x)^{(\alpha)}$ with one parameter normalizer
          *
          * Internal computation can be done in multiple precision
          */
         static void P0(Eigen::Ref<internal::Matrix> ip0, const internal::MHDFloat alpha, const internal::Array& igrid, NormalizerA norm);

         /**
          * @brief Compute three term recurrence for $P_n(x)^{(\alpha,\beta)}$ with two parameter normalizer
          *
          * Internal computation can be done in multiple precision
          */
         static void Pn(Eigen::Ref<internal::Matrix> ipn, const int n, const internal::MHDFloat alpha, const internal::MHDFloat beta, const Eigen::Ref<const internal::Matrix>& ipn_1, const Eigen::Ref<const internal::Matrix>& ipn_2, const internal::Array& igrid, NormalizerNAB norm);

         /**
          * @brief Compute \f$P_1^{(\alpha,\beta)} (x)\f$
          *
          * Internal computation can be done in multiple precision
          */
         static void P1(Eigen::Ref<internal::Matrix> ip1, const internal::MHDFloat alpha, const internal::MHDFloat beta, const Eigen::Ref<const internal::Matrix>& ip0, const internal::Array& igrid, NormalizerAB norm);

         /**
          * @brief Compute \f$P_0^{(\alpha,\beta)} (x)\f$
          *
          * Internal computation can be done in multiple precision
          */
         static void P0(Eigen::Ref<internal::Matrix> ip0, const internal::MHDFloat alpha, const internal::MHDFloat beta, const internal::Array& igrid, NormalizerAB norm);

      private:
         /**
          * @brief Constructor
          */
         ThreeTermRecurrence();

         /**
          * @brief Destructor
          */
         ~ThreeTermRecurrence();

   };
}
}

#endif // THREETERMRECURRENCE_HPP
