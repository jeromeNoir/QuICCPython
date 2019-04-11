/** 
 * @file Precision.cpp
 * @brief Implementation of precision related constants and routines
 * @author Philippe Marti \<philippe.marti@colorado.edu\>
 */

// System includes
//

// External includes
//

// Class include
//
#include "Base/Precision.hpp"

// Project includes
//

namespace QuICC {
   const internal::MHDFloat Precision::PI = precision::acos(MHD_MP(-1.0));

}
