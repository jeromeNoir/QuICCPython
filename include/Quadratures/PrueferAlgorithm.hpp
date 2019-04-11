/**
 * @file PrueferAlgorithm.hpp
 * @brief Implementation of a Pruefer algorithm to compute a quadrature rule 
 * @author Philippe Marti \<philippe.marti@colorado.edu\>
 */

#ifndef PRUEFERALGORITHM_HPP
#define PRUEFERALGORITHM_HPP

// System includes
//

// External includes
//

// Project includes
//
#include "Base/Precision.hpp"

namespace QuICC {

   /**
    * @brief Implementation of an iterative quadrature rule computation.
    *
    * This implementation relies (in general) on a coarse Runge-Kutta solver to find an initial guess. Newton
    * iterations are then used to reach high accuracy. It does not require any linear algebra solvers, removing the
    * issues coming from the lack of special matrix solvers in Eigen. Multiple precision values are obtained through
    * the MPFR wrapper.
    *
    * The implemented algorithms are based on the paper by Glaser, Liu & Rohklin, 2007. "A fast algorithm for the 
    * calculation of the roots of special functions"
    */
   class PrueferAlgorithm
   {
      public:
         /**
          * @brief Number of Newton iterations in refinement loop
          */
         static const int NEWTON_ITERATIONS;

         /**
          * @brief Maximum order in the Taylor expansion
          */
         static const int TAYLOR_ORDER;

         /**
          * @brief Compute the Taylor expansion
          *
          * @param taylor  Storage for the taylor expansion
          * @param size    Grid size
          * @param u_0     Zeroth order Taylor expansion
          * @param u_1     First order Taylor expansion
          * @param xi_1    Grid value
          */
         template <typename TRule> static void computeTaylor(internal::Array& taylor, const int size, const internal::MHDFloat u_0, const internal::MHDFloat u_1, const internal::MHDFloat xi_1);

         /**
          * @brief Refine node value through Newton iteration
          *
          * @param i Index of the node
          */
         static void refineNode(internal::Array& grid, internal::Array& weights, const int i, const internal::Array& taylor);

         /**
          * @brief Sort the grid and weights from the quadrature
          */
         static void sortQuadrature(internal::Array& grid, internal::Array& weights);

      protected:

      private:
         /**
          * @brief Empty constructor
          */
         PrueferAlgorithm();

         /**
          * @brief Empty destructor
          */
         virtual ~PrueferAlgorithm();
   };

   template <typename TRule> void PrueferAlgorithm::computeTaylor(internal::Array& taylor, const int size, const internal::MHDFloat u_0, const internal::MHDFloat u_1, const internal::MHDFloat xi_1)
   {
      // Make sure to reset to zero
      taylor.setConstant(MHD_MP(0.0));

      // Fill zeroth and first order
      taylor(0) = u_0;
      taylor(1) = u_1;

      for(int k = 0; k < taylor.size()-2; k++)
      {
         // First term 
         taylor(k+2) = -(static_cast<internal::MHDFloat>(k)*TRule::p(xi_1,1) + TRule::q(xi_1,0))*taylor(k+1);

         // Second term 
         taylor(k+2) -= ((static_cast<internal::MHDFloat>(k*(k-1))/MHD_MP(2.0))*TRule::p(xi_1,2) + static_cast<internal::MHDFloat>(k)*TRule::q(xi_1,1) + TRule::r(size,0))*taylor(k);

         // Third term (if applicable)
         if(k > 0)
         {
            taylor(k+2) -= ((static_cast<internal::MHDFloat>(k*(k-1))/MHD_MP(2.0))*TRule::q(xi_1,2) + static_cast<internal::MHDFloat>(k)*TRule::r(size,1))*taylor(k-1);
         }

         // Fourth term (if applicable)
         if(k > 1)
         {
            taylor(k+2) -= ((static_cast<internal::MHDFloat>(k*(k-1))/MHD_MP(2.0))*TRule::r(size,2))*taylor(k-2);
         }

         taylor(k+2) /= TRule::p(xi_1,0);
      }
   }

}

#endif // PRUEFERALGORITHM_HPP
