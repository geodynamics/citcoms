// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <portinfo>
#include <iostream>

#include "Exchanger.h"
#include "FineGridExchanger.h"



FineGridExchanger::FineGridExchanger(const All_variables *E) {
    cout << "in FineGridExchanger::FineGridExchanger" << std::endl;
}



FineGridExchanger::~FineGridExchanger() {
    cout << "in FineGridExchanger::~FineGridExchanger" << std::endl;
}



void FineGridExchanger::setup() {
    cout << "in FineGridExchanger::setup" << std::endl;
}


void FineGridExchanger::gather() {
    cout << "in FineGridExchanger::gather" << std::endl;
}



void FineGridExchanger::distribute() {
    cout << "in FineGridExchanger::distribute" << std::endl;
}



void FineGridExchanger::interpretate() {
    cout << "in FineGridExchanger::interpretate" << std::endl;
}




void FineGridExchanger::impose_bc() {
    cout << "in FineGridExchanger::impose_bc" << std::endl;

}



// version
// $Id: FineGridExchanger.cc,v 1.1 2003/09/06 23:44:22 tan2 Exp $

// End of file
