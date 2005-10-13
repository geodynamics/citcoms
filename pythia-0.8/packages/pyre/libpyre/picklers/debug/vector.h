// -*- C++ -*-
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                              Michael A.G. Aivazis
//                       California Institute of Technology
//                       (C) 1998-2005  All Rights Reserved
//
// <LicenseText>
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyre_picklers_debug_vector_h)
#define pyre_picklers_debug_vector_h

#include <string>
#include <sstream>

// declare in the right namespace

namespace pyre {
    namespace picklers {
        namespace debug {
            template <typename vector_t> std::string vector(const vector_t & p);
        }
    }
}


// the definition

template <typename vector_t> 
std::string pyre::picklers::debug::vector(const vector_t & v) {

    std::ostringstream out;

    out << "vector@{" << &v << "} = {";
    for (size_t i = 0; i < v.size(); ++i) {
	out << " " << v(i);
    }
    out << " }";

    return out.str();
}

#endif

// version
// $Id: vector.h,v 1.1.1.1 2005/03/08 16:13:50 aivazis Exp $

// End of file
