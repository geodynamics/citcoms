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

#include "pyre/manipulators/manip3.h"

#include <string>
#include <iostream>

using namespace pyre::manipulators;

inline std::ostream & 
mgamanip_set(std::ostream & s, const std::string m1, const std::string m2, const std::string m3) {
    s << "mgaset: " << m1 << ": " << m2 << " " << m3;
    return s;
}

inline manip_3<std::ostream, const std::string, const std::string, const std::string> 
mgaset(const std::string m1, const std::string m2, const std::string m3) {
    return manip_3<std::ostream, const std::string, const std::string, const std::string>
        (mgamanip_set, m1, m2, m3);
}

int main() {

    std::cout << mgaset("mga", "Hello", "world!") << std::endl;
    return 0;
}


// version
// $Id: manip3.cc,v 1.1.1.1 2005/03/08 16:13:50 aivazis Exp $

//  End of file 
