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

#include "pyre/containers/Vector.h"
#include "pyre/picklers/debug/vector.h"


int main()
{

    const size_t dim = 3;
    typedef pyre::containers::Vector<double> vector_t;

    vector_t v1(dim);
    for (size_t i = 0; i < dim; ++i) {
        v1(i) = i;
    }
    vector_t v2 = v1;

    vector_t v3(dim);
    v3 = v1;

    std::cout << pyre::picklers::debug::vector(v1) << std::endl;
    std::cout << pyre::picklers::debug::vector(v2) << std::endl;
    std::cout << pyre::picklers::debug::vector(v3) << std::endl;

    return 0;
}

// version
// $Id: vector.cc,v 1.1.1.1 2005/03/08 16:13:50 aivazis Exp $

// End of file
