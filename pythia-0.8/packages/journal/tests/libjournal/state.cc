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
#include "journal/info.h"
#include <iostream>

int main() {

    std::cout << " --- test: state" << std::endl;

    journal::info_t info("test");
    std::cout << info.name() << "(1): state=" << info.state() << std::endl;

    journal::info_t info2("test");
    info2.activate();
    std::cout << info.name() << "(1): state=" << info.state() << std::endl;
    std::cout << info2.name() << "(2): state=" << info2.state() << std::endl;

    std::cout << " --- test: state done" << std::endl;

    return 0;
}

// version
// $Id: state.cc,v 1.1.1.1 2005/03/08 16:13:54 aivazis Exp $

// End of file 
