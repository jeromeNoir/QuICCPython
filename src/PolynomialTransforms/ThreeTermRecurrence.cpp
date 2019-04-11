/** 
 * @file ThreeTermRecurrence.cpp
 * @brief Source of the general three term recurrence implementation
 * @author Philippe Marti \<philippe.marti@colorado.edu\>
 */

// System includes
//

// External includes
//

// Class include
//
#include "PolynomialTransforms/ThreeTermRecurrence.hpp"

// Project includes
//

namespace QuICC {

namespace Polynomial {

   void ThreeTermRecurrence::Pn(Eigen::Ref<internal::Matrix> ipn, const int n, const Eigen::Ref<const internal::Matrix>& ipn_1, const Eigen::Ref<const internal::Matrix>& ipn_2, const internal::Array& igrid, NormalizerNC norm)
   {
      internal::MHDFloat dn = internal::MHDFloat(n);
      internal::Array cs = norm(dn);

      ipn.array() = cs(0)*ipn_2.array();
      ipn.array() += (cs(1)*igrid.array() + cs(2))*ipn_1.array();
      ipn.array() *= cs(3);
   }

   void ThreeTermRecurrence::P1(Eigen::Ref<internal::Matrix> ip1, const Eigen::Ref<const internal::Matrix>& ip0, const internal::Array& igrid, NormalizerC norm)
   {
      internal::Array cs = norm();

      ip1.array() = (cs(0)*igrid.array() + cs(1))*ip0.array();
      ip1.array() *= cs(2);
   }

   void ThreeTermRecurrence::P0(Eigen::Ref<internal::Matrix> ip0, const internal::Array& igrid, NormalizerC norm)
   {
      internal::Array cs = norm();

      ip0.setConstant(cs(0));
   }

   void ThreeTermRecurrence::Pn(Eigen::Ref<internal::Matrix> ipn, const int n, const internal::MHDFloat alpha, const Eigen::Ref<const internal::Matrix>& ipn_1, const Eigen::Ref<const internal::Matrix>& ipn_2, const internal::Array& igrid, NormalizerNA norm)
   {
      internal::MHDFloat dn = internal::MHDFloat(n);
      internal::Array cs = norm(dn, alpha);

      ipn.array() = cs(0)*ipn_2.array();
      ipn.array() += (cs(1)*igrid.array() + cs(2))*ipn_1.array();
      ipn.array() *= cs(3);
   }

   void ThreeTermRecurrence::P1(Eigen::Ref<internal::Matrix> ip1, const internal::MHDFloat alpha, const Eigen::Ref<const internal::Matrix>& ip0, const internal::Array& igrid, NormalizerA norm)
   {
      internal::Array cs = norm(alpha);

      ip1.array() = (cs(0)*igrid.array() + cs(1))*ip0.array();
      ip1.array() *= cs(2);
   }

   void ThreeTermRecurrence::P0(Eigen::Ref<internal::Matrix> ip0, const internal::MHDFloat alpha, const internal::Array& igrid, NormalizerA norm)
   {
      internal::Array cs = norm(alpha);

      ip0.setConstant(cs(0));
   }

   void ThreeTermRecurrence::Pn(Eigen::Ref<internal::Matrix> ipn, const int n, const internal::MHDFloat alpha, const internal::MHDFloat beta, const Eigen::Ref<const internal::Matrix>& ipn_1, const Eigen::Ref<const internal::Matrix>& ipn_2, const internal::Array& igrid, NormalizerNAB norm)
   {
      internal::MHDFloat dn = internal::MHDFloat(n);
      internal::Array cs = norm(dn, alpha, beta);

      ipn.array() = cs(0)*ipn_2.array();
      ipn.array() += (cs(1)*igrid.array() + cs(2))*ipn_1.array();
      ipn.array() *= cs(3);
   }

   void ThreeTermRecurrence::P1(Eigen::Ref<internal::Matrix> ip1, const internal::MHDFloat alpha, const internal::MHDFloat beta, const Eigen::Ref<const internal::Matrix>& ip0, const internal::Array& igrid, NormalizerAB norm)
   {
      internal::Array cs = norm(alpha, beta);

      ip1.array() = (cs(0)*igrid.array() + cs(1))*ip0.array();
      ip1.array() *= cs(2);
   }

   void ThreeTermRecurrence::P0(Eigen::Ref<internal::Matrix> ip0, const internal::MHDFloat alpha, const internal::MHDFloat beta, const internal::Array& igrid, NormalizerAB norm)
   {
      internal::Array cs = norm(alpha, beta);

      ip0.setConstant(cs(0));
   }

   ThreeTermRecurrence::ThreeTermRecurrence()
   {
   }

   ThreeTermRecurrence::~ThreeTermRecurrence()
   {
   }

}
}
