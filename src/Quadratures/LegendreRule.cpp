/** 
 * @file LegendreRule.cpp
 * @brief Source of the Legendre quadrature
 * @author Philippe Marti \<philippe.marti@colorado.edu\>
 */

// System includes
//

// External includes
//

// Class include
//
#include "Quadratures/LegendreRule.hpp"
#include "Quadratures/PrueferAlgorithm.hpp"

// Project includes
//

namespace QuICC {

   void LegendreRule::computeQuadrature(Array& grid, Array& weights, const int size)
   {
      // Internal grid and weights arrays
      internal::Array   igrid;
      internal::Array   iweights;

      LegendreRule::computeQuadrature(grid, weights, igrid, iweights, size);
   }

   void LegendreRule::computeQuadrature(Array& grid, Array& weights, internal::Array& igrid, internal::Array& iweights, const int size)
   {
      // Internal grid and weights arrays
      igrid.resize(size);
      igrid.setConstant(MHD_MP(0.0));
      iweights.resize(size);
      iweights.setConstant(MHD_MP(0.0));

      internal::Array   taylor(std::min(size+1,PrueferAlgorithm::TAYLOR_ORDER+1));

      // Need to treat differently odd or even sizes
      if(size % 2 == 0)
      {
         // Start from extremum value 
         igrid(0) = MHD_MP(0.0);
         igrid(1) = LegendreRule::estimateNode(0, size);
         // Compute accurate node and derivative
         PrueferAlgorithm::computeTaylor<LegendreRule>(taylor, size, LegendreRule::zeroPoly(size), MHD_MP(0.0), igrid(0));
         PrueferAlgorithm::refineNode(igrid, iweights, 1, taylor);

         // Set obtained node as first node and its derivative
         igrid(0) = igrid(1);
         iweights(0) = iweights(1);

         // Start filling opposite end
         igrid(size-1) = -igrid(0);
         iweights(size-1) = -iweights(0);

      } else
      {
         // 0 is a node
         igrid(0) = MHD_MP(0.0);
         iweights(0) = LegendreRule::zeroDiff(size);
      }

      // Compute grid
      for(int i = 1; i < size/2+1; i++)
      {
         // Estimate node position with formulae
         igrid(i) = LegendreRule::estimateNode(i - size%2, size);
         PrueferAlgorithm::computeTaylor<LegendreRule>(taylor, size, MHD_MP(0.0), iweights(i-1), igrid(i-1));
         PrueferAlgorithm::refineNode(igrid, iweights, i, taylor);

         // If solution is too far from estimate, redo full loop starting from solution
         // This should only be required for "small" number of grid points (estimate is asymptotic formulae)
         if(precision::abs((igrid(i) - LegendreRule::estimateNode(i - size%2, size))/igrid(i)) > 1.0e-8)
         {
            PrueferAlgorithm::computeTaylor<LegendreRule>(taylor, size, MHD_MP(0.0), iweights(i-1), igrid(i-1));
            PrueferAlgorithm::refineNode(igrid, iweights, i, taylor);
         }

         // Fill other half of symmetric nodes
         igrid(size - 1 - i + (size%2)) = -igrid(i);
         iweights(size - 1 - i + (size%2)) = iweights(i);
      }

      // Convert derivative to weights
      iweights = MHD_MP(2.0)*(MHD_MP(1.0)-igrid.array().square()).array().inverse()*iweights.array().inverse().square();

      // Sort the grid and weights
      PrueferAlgorithm::sortQuadrature(igrid, iweights);

      // Copy internal precision values into input arrays
      grid = Precision::cast(igrid);
      weights = Precision::cast(iweights);
   }

   internal::MHDFloat   LegendreRule::p(const internal::MHDFloat xi, const int diff)
   {
      // Safety asserts
      assert(diff >= 0);

      // Get p polynomial
      if(diff == 0)
      {
         return MHD_MP(1.0)-xi*xi;

      // Get first derivative of p polynomial
      } else if(diff == 1)
      {
         return MHD_MP(-2.0)*xi;

      // Get second derivative of p polynomial
      } else if(diff == 2)
      {
         return MHD_MP(-2.0);

      } else
      {
         return MHD_MP(0.0);
      }
   }

   internal::MHDFloat   LegendreRule::q(const internal::MHDFloat xi, const int diff)
   {
      // Safety asserts
      assert(diff >= 0);

      // Get q polynomial
      if(diff == 0)
      {
         return MHD_MP(-2.0)*xi;

      // Get first derivative of q polynomial
      } else if(diff == 1)
      {
         return MHD_MP(-2.0);

      // Get second derivative of q polynomial
      } else if(diff == 2)
      {
         return MHD_MP(0.0);

      } else
      {
         return MHD_MP(0.0);
      }
   }

   internal::MHDFloat   LegendreRule::r(const int size, const int diff)
   {
      // Safety asserts
      assert(diff >= 0);

      // Get r polynomial
      if(diff == 0)
      {
         return static_cast<internal::MHDFloat>(size*(size+1));

      // Get first derivative of r polynomial
      } else if(diff == 1)
      {
         return MHD_MP(0.0);

      // Get second derivative of r polynomial
      } else if(diff == 2)
      {
         return MHD_MP(0.0);

      } else
      {
         return MHD_MP(0.0);
      }
   }

   internal::MHDFloat LegendreRule::estimateNode(const int k, const int size)
   {
      // Storage for the node estimate
      internal::MHDFloat   x;

      // Cast grid size to floating value
      internal::MHDFloat   rN = static_cast<internal::MHDFloat>(size);

      // Get theta value
      internal::MHDFloat   th = static_cast<internal::MHDFloat>(4*((size/2)-k)-1)/(MHD_MP(4.0)*rN+MHD_MP(2.0))*Precision::PI;

      x = (MHD_MP(1.0) - (rN-MHD_MP(1.0))/(8*rN*rN*rN) - MHD_MP(1.0)/(MHD_MP(384.0)*rN*rN*rN*rN)*(MHD_MP(39.0)-MHD_MP(28.0)/(precision::sin(th)*precision::sin(th))))*precision::cos(th);

      return x;
   }

   internal::MHDFloat LegendreRule::zeroPoly(const int size)
   {
      // Initialise start value of recurrence
      internal::MHDFloat p = MHD_MP(1.0);

      for(int i = 0; i < size/2; i++)
      {
         p *= -static_cast<internal::MHDFloat>(2*i+1)/static_cast<internal::MHDFloat>((2*i+1) + 1);
      }

      return p;
   }

   internal::MHDFloat LegendreRule::zeroDiff(const int size)
   {
      // Initialise start value of recurrence
      internal::MHDFloat p = MHD_MP(1.0);

      // Initialise start value of recurrence
      internal::MHDFloat dp = MHD_MP(1.0);

      for(int i = 0; i < size/2; i++)
      {
         p *= -static_cast<internal::MHDFloat>(2*i+1)/static_cast<internal::MHDFloat>((2*i+1) + 1);

         dp *= -static_cast<internal::MHDFloat>(2*i+2)/static_cast<internal::MHDFloat>((2*i+2) + 1);
         dp += static_cast<internal::MHDFloat>(2*(2*i+2)+1)/static_cast<internal::MHDFloat>((2*i+2) + 1)*p;
      }

      return dp;
   }

}
