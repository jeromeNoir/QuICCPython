/** 
 * @file LegendreRule.hpp
 * @brief Implementation of a Legendre quadrature rule
 * @author Philippe Marti \<philippe.marti@colorado.edu\>
 */

#ifndef LEGENDRERULE_HPP
#define LEGENDRERULE_HPP

// System includes
//

// External includes
//

// Project includes
//
#include "Base/Precision.hpp"

namespace QuICC {

   /**
    * @brief Implementation of a Legendre quadrature rule
    */
   class LegendreRule
   {
      public:
         /**
          * @brief Compute the quadrature and return standard precision values
          */
         static void computeQuadrature(Array& grid, Array& weights, const int size);

         /**
          * @brief Compute the quadrature and return standard and internal precision values
          */
         static void computeQuadrature(Array& grid, Array& weights, internal::Array& igrid, internal::Array& iweights, const int size);

         /**
          * @brief Get p polynomial
          *
          * @param xi   Grid value
          * @param diff Order of the derivative
          */
         static internal::MHDFloat   p(const internal::MHDFloat xi, const int diff);

         /**
          * @brief Get q polynomial
          *
          * @param xi   Grid value
          * @param diff Order of the derivative
          */
         static internal::MHDFloat   q(const internal::MHDFloat xi, const int diff);

         /**
          * @brief Get r polynomial
          *
          * @param size Size of the grid
          * @param diff Order of the derivative
          */
         static internal::MHDFloat   r(const int size, const int diff);

         /**
          * @brief Node estimate
          *
          * @param k Index of the node to estimate
          */
         static internal::MHDFloat   estimateNode(const int k, const int size);

         /**
          * @brief Compute polynomial value at 0
          */
         static internal::MHDFloat   zeroPoly(const int size);

         /**
          * @brief Compute first derivative value at 0
          */
         static internal::MHDFloat   zeroDiff(const int size);
         
      protected:

      private:
         /**
          * @brief Empty constructor
          */
         LegendreRule();

         /**
          * @brief Empty destructor
          */
         virtual ~LegendreRule();
   };

}

#endif // LEGENDRERULE_HPP
