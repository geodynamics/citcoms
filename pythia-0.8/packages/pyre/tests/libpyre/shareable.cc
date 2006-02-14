// -*- C++ -*-
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                              Michael A.G. Aivazis
//                       California Institute of Technology
//                       (C) 1998-2005 All Rights Reserved
//
// <LicenseText>
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <iostream>

#include "journal/journal.h"
#include "pyre/shareable.h"

// make a reference counted Point

struct Point : public pyre::memory::Shareable {
public:

    Point(double x, double y, double z):
        Shareable(),
	x(x), y(y), z(z)
	{}

    double x, y, z;
};

typedef pyre::memory::PtrShareable<Point> PointHandle;


int main()
{
    journal::debug_t debug("pyre.memory");
    debug.activate();

    // ctor
    Point p(0,0,0);
    Point q = p;
    Point r(2,2,2);
    r = p;

    // handle
    PointHandle p_ptr(&p);

    std::cout << "p = {" << p_ptr->x << "," << p_ptr->y << "," << p_ptr->z  << "}" << std::endl;
    std::cout << "p = {" << p.x << "," << p.y << "," << p.z  << "}" << std::endl;
    std::cout << "q = {" << q.x << "," << q.y << "," << q.z  << "}" << std::endl;
    std::cout << "r = {" << r.x << "," << r.y << "," << r.z  << "}" << std::endl;

    return 0;
}

// version
// $Id: shareable.cc,v 1.1.1.1 2005/03/08 16:13:50 aivazis Exp $

// End of file
