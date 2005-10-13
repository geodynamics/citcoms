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

#if !defined(pyre_picklers_debug_tuple_h)
#define pyre_picklers_debug_tuple_h

#include <string>
#include <sstream>

// declare in the right namespace

namespace pyre {
    namespace picklers {
        namespace debug {
            template <typename tuple_t> std::string tuple(const tuple_t & p);
        }
    }
}


// the definition

template <typename tuple_t> 
std::string pyre::picklers::debug::tuple(const tuple_t & t) {

    std::ostringstream out;

    out << "tuple@{" << &t << "} = {";
    for (size_t i = 0; i < t.length; ++i) {
	out << " " << t(i);
    }
    out << " }";

    return out.str();
}

#endif

// version
// $Id: tuple.h,v 1.1.1.1 2005/03/08 16:13:50 aivazis Exp $

// End of file
