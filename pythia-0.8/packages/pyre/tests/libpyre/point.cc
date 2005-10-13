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
#include <portinfo>
#include <iostream>

#include "journal/journal.h"
#include "../src/geometry/Point.h"
#include "../src/picklers/debug/point.h"


int main()
{
    journal::debug_t("pyre.geometry.Point").activate();

    const size_t dim = 3;
    typedef pyre::geometry::Point<dim, double> point_t;

    point_t origin;
    origin = 0.0, 0.0, 0.0;

    std::cout << pyre::picklers::debug::point(origin) << std::endl;

    return 0;
}

// version
// $Id: point.cc,v 1.1.1.1 2005/03/08 16:13:50 aivazis Exp $

// End of file
