// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include "Boundary.h"

Boundary::Boundary(const int n):
    size(n),
    connectivity(new int[size]) {

    for(int i=0; i<dim; i++)
	X[i] = std::auto_ptr<double>(new double[size]);
}


Boundary::~Boundary() {};


void Boundary::printConnectivity() const {
    for(int i=0; i<dim; i++) {
	std::cout << "dimension: " << i << std::endl;
	int *c = connectivity.get();
	for(int j=0; j<size; j++)
	    std::cout << "    " << j << ":  " << c[j] << std::endl;
    }
}


// version
// $Id: Boundary.cc,v 1.1 2003/09/09 02:35:22 tan2 Exp $

// End of file
