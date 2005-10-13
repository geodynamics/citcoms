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
#include "journal/diagnostics.h"
#include <iostream>

int main() {

    journal::info_t info("test");
    info.activate();
    std::cout << info.name() << ": state=" << info.state() << std::endl;

    info << journal::at(__HERE__) << "Hello world!" << journal::endl;

    return 0;
}

// version
// $Id: diagnostics.cc,v 1.1.1.1 2005/03/08 16:13:54 aivazis Exp $

// End of file 
