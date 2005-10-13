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

#define JOURNAL_NON_TEMPLATED_MANIPULATORS
#include "journal/info.h"
#include <iostream>

int main() {

    journal::info_t info("test");
    info.activate();
    std::cout << info.name() << ": state=" << info.state() << std::endl;

    info
        << journal::at(__HERE__) << journal::set("key", "value")
        << "Hello world!"
        << journal::endl;

    return 0;
}

// version
// $Id: info.cc,v 1.1.1.1 2005/03/08 16:13:54 aivazis Exp $

// End of file 
