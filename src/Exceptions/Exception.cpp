/** 
 * @file Exception.cpp
 * @brief Definitions of the Exception methods.
 * @author Philippe Marti \<philippe.marti@colorado.edu\>
 */

// System includes
//

// External includes
//

// Class include
//
#include "Exceptions/Exception.hpp"

// Project includes
//

namespace QuICC {

   Exception::Exception(const std::string& msg)
      : std::runtime_error(msg)   
   {
   }

   Exception::~Exception() throw()
   {
   }

}
