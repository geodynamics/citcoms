// -*- C++ -*-
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                       (C) 1998-2005  All Rights Reserved
//
// <LicenseText>
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <iostream>

#include "pyre/containers/Tuple.h"
#include "pyre/picklers/debug/tuple.h"


int main()
{

    const size_t dim = 3;
    typedef pyre::containers::Tuple<double, dim> tuple_t;

    tuple_t t1;
    t1 = 1,2,3;

    std::cout << pyre::picklers::debug::tuple(t1) << std::endl;

    return 0;
}

// version
// $Id: tuple.cc,v 1.1.1.1 2005/03/08 16:13:50 aivazis Exp $

// End of file
