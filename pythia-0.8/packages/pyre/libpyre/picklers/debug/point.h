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

#if !defined(pyre_picklers_debug_point_h)
#define pyre_picklers_debug_point_h

#include <string>
#include <sstream>

// declare in the right namespace

namespace pyre {
    namespace picklers {
        namespace debug {
            template <typename point_t> std::string point(const point_t & p);
        }
    }
}


// the definition

template <typename point_t> 
std::string pyre::picklers::debug::point(const point_t & p) {

    std::ostringstream out;

    out << "point@{" << &p << "} = (";
    for (size_t i = 0; i < p.dimension; ++i) {
	out << " " << p(i);
    }
    out << " )";

    return out.str();
}

#endif

// version
// $Id: point.h,v 1.1.1.1 2005/03/08 16:13:51 aivazis Exp $

// End of file
