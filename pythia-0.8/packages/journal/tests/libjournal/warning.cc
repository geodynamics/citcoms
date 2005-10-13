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
#include "journal/warning.h"
#include <iostream>

int main() {

    journal::warning_t warning("test");
    std::cout << warning.name() << ": state=" << warning.state() << std::endl;

    warning << journal::at(__HERE__) << "Hello world!" << journal::endl;

    return 0;
}

// version
// $Id: warning.cc,v 1.1.1.1 2005/03/08 16:13:54 aivazis Exp $

// End of file 
