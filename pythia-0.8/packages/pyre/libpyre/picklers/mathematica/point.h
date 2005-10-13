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

#if !defined(pyre_picklers_mathematica_debug_h)
#define pyre_picklers_mathematica_debug_h

#include <string>
#include <sstream>

// declare in the right namespace

namespace plexus {
    namespace picklers {
        namespace mathematica {
            template <typename point_t> std::string point(const point_t & p);
        }
    }
}


// the definition

template <typename point_t> 
std::string plexus::picklers::mathematica::point(const point_t & p) {

    std::ostringstream out;

    out << "Point[{";

    for (int axis = 0; axis < p.dimension - 1; ++axis) {
	out << p(axis) << ",";
    }
    out << p(p.dimension - 1);

    out << "}]";

    return out.str();
}


#endif

// version
// $Id: point.h,v 1.1.1.1 2005/03/08 16:13:51 aivazis Exp $

// End of file
