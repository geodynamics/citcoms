// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <iostream>

#include "global_defs.h"
#include "Boundary.h"

Boundary::Boundary(const int n):
    size(n),
    connectivity_(new int[size]) {

    // use auto_ptr for exception-proof
    for(int i=0; i<dim; i++)
	X_[i] = std::auto_ptr<double>(new double[size]);

    // setup traditional pointer for convenience
    connectivity = connectivity_.get();
    for(int i=0; i<dim; i++)
	X[i] = X_[i].get();

}



Boundary::~Boundary() {};



void Boundary::init(const All_variables *E) {

    // test
    for(int j=0; j<size; j++)
	connectivity[j] = j;

    for(int i=0; i<dim; i++)
	for(int j=0; j<size; j++) {
	    X[i][j] = i+j;
	}
}



void Boundary::map(const All_variables *E) {

}



void Boundary::printConnectivity() const {
    for(int i=0; i<dim; i++) {
	std::cout << "dimension: " << i << std::endl;
	int *c = connectivity;
	for(int j=0; j<size; j++)
	    std::cout << "    " << j << ":  " << c[j] << std::endl;
    }
}


// version
// $Id: Boundary.cc,v 1.2 2003/09/09 18:25:31 tan2 Exp $

// End of file
