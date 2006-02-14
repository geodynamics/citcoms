// -*- C++ -*-
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2005 All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 

#include "pyre/manipulators/manip1.h"

#include <string>
#include <iostream>

using namespace pyre::manipulators;

inline std::ostream & mgamanip_set(std::ostream & s, const std::string msg) {
    s << "mgaset: " << msg;
    return s;
}

inline manip_1<std::ostream, const std::string> 
mgaset(const std::string msg) {
    return manip_1<std::ostream, const std::string>(mgamanip_set, msg);
}

int main() {

    std::cout << mgaset("Hello world!") << std::endl;
    return 0;
}


// version
// $Id: manip1.cc,v 1.1.1.1 2005/03/08 16:13:50 aivazis Exp $

//  End of file 
