/** 
 * @file PrueferAlgorithm.cpp
 * @brief Source of the implementation of the Pruefer algorithm for the computation of a quadrature rule
 * @author Philippe Marti \<philippe.marti@colorado.edu\>
 */

// Configuration includes
//

// System includes
//
#include <map>

// External includes
//

// Interfaces includes
//

// Class include
//
#include "Quadratures/PrueferAlgorithm.hpp"

// Project includes
//
#include "Exceptions/Exception.hpp"

namespace QuICC {

#ifdef QUICC_MULTPRECISION
   const int PrueferAlgorithm::NEWTON_ITERATIONS = 10;

   const int PrueferAlgorithm::TAYLOR_ORDER = 60;
#else
   const int PrueferAlgorithm::NEWTON_ITERATIONS = 5;

   const int PrueferAlgorithm::TAYLOR_ORDER = 30;
#endif //QUICC_MULTPRECISION

   void PrueferAlgorithm::refineNode(internal::Array& grid, internal::Array& weights, const int i, const internal::Array& taylor)
   {
      // Storage for (x_{k+1} - x_{k})
      internal::MHDFloat   h;
      // Storage for the poylnomial value
      internal::MHDFloat   f;

      // Compute Newton refinement iterations
      for(int n = 0; n < PrueferAlgorithm::NEWTON_ITERATIONS; n++)
      {
         // Initialise starting values
         h = MHD_MP(1.0);
         f = MHD_MP(0.0);
         weights(i) = MHD_MP(0.0);

         // Loop over all Taylor coefficients
         for(int k = 0; k < taylor.size()-1; k++)
         {
            // Add next order to function value
            f += taylor(k)*h;

            // Add next order to derivative value
            weights(i) += taylor(k+1)*h;

            // Increment the monomial (including factorial part)
            h *= (grid(i) - grid(i-1))/static_cast<internal::MHDFloat>(k+1);
         }

         // Add last order to function value
         f += taylor(taylor.size()-1)*h;

         // update the Newton iteration value
         grid(i) = grid(i) - f/weights(i);
      }

   }

   void PrueferAlgorithm::sortQuadrature(internal::Array& grid, internal::Array& weights)
   {
      // Safety assert
      assert(grid.size() == weights.size());

      // Create a map to sort elements
      std::map<internal::MHDFloat, internal::MHDFloat> sorter;

      // fill map with grid/weights pairs
      for(int i = 0; i < grid.size(); ++i)
      {
         sorter.insert(std::make_pair(grid(i), weights(i)));
      }

      // Check that no point got lost
      if(sorter.size() != static_cast<size_t>(grid.size()))
      {
         throw Exception("PrueferAlgorithm::sortGrid: Lost grid points during sorting!");
      }

      std::map<internal::MHDFloat, internal::MHDFloat>::const_iterator  it;

      // Replace grid point values with reordered version
      int i = 0;
      for(it = sorter.begin(); it != sorter.end(); ++it, ++i)
      {
         grid(i) = it->first;
         weights(i) = it->second;
      }
   }
}
