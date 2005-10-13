// -*- C++ -*-
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                             Michael A.G. Aivazis
//                      California Institute of Technology
//                      (C) 1998-2005  All Rights Reserved
//
// <LicenseText>
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "journal/debug.h"
#include <iostream>

int main() {

    journal::debug_t debug("test");
    debug.activate();
    std::cout << debug.name() << ": state=" << debug.state() << std::endl;

    debug << journal::at(__HERE__) << "Hello world!" << journal::endl;

    return 0;
}

// version
// $Id: debug.cc,v 1.1.1.1 2005/03/08 16:13:54 aivazis Exp $

// End of file 
