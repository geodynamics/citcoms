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

    debug 
        << journal::at(__HERE__) 
        << "Hello world!" 
        << journal::newline
        << "Here is an integer:" << 0xdeadbeef
        << journal::newline
        << std::hex
        << "Here it is again: 0x" << 0xdeadbeef
        << journal::endl;

    return 0;
}

// version
// $Id: control.cc,v 1.1.1.1 2005/03/08 16:13:54 aivazis Exp $

// End of file 
