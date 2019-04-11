/** 
 * @file SharedPtrMacro.h
 * @brief Preprocessor macros used to setup the shared pointer implementation depending on CMake setup.
 * @author Philippe Marti \<philippe.marti@colorado.edu\>
 */

#ifndef SHAREDPTRMACRO_H
#define SHAREDPTRMACRO_H

// Boost Version
#ifdef QUICC_SMARTPTR_BOOST
   // Include the right header
   #include <boost/shared_ptr.hpp>
   #include <boost/tuple/tuple.hpp>
   /**
    * @def SharedPtrMacro
    * Macro allowing to use different implementations of the shared_ptr smart pointer.
    * Sets up the Boost version.
    */
   #define SharedPtrMacro boost::shared_ptr
   #define StaticPtrCast boost::static_pointer_cast
   #define TupleMacro boost
#endif //QUICC_SMARTPTR_BOOST

// TR1 Version
#ifdef QUICC_SMARTPTR_TR1
   // Include the right header
   #include <tr1/memory>
   #include <tr1/tuple>
   /**
    * @def SharedPtrMacro
    * Macro allowing to use different implementations of the shared_ptr smart pointer.
    * Sets up the TR1 version.
    */
   #define SharedPtrMacro std::tr1::shared_ptr
   #define StaticPtrCast std::tr1::static_pointer_cast
   #define TupleMacro std::tr1 
#endif //QUICC_SMARTPTR_TR1

// C++0x Version
#ifdef QUICC_SMARTPTR_CXX0X
   // Include the right header
   #include <memory>
   #include <tuple>
   /**
    * @def SharedPtrMacro
    * Macro allowing to use different implementations of the shared_ptr smart pointer.
    * Sets up the C++0x version.
    */
   #define SharedPtrMacro std::shared_ptr
   #define StaticPtrCast std::static_pointer_cast
   #define TupleMacro std   
#endif //QUICC_SMARTPTR_CXX0X

#endif // SHAREDPTRMACRO_H
