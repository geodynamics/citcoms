// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#if !defined(pyCitcom_Boundary_h)
#define pyCitcom_Boundary_h

#include <memory>
#include <iostream>

class Boundary {
public:
    static const int dim = 3;  // spatial dimension

    const int size;            // # of boundary nodes
    std::auto_ptr<int> connectivity;
    std::auto_ptr<double> X[dim];    // coordinate

    explicit Boundary(const int n);
    ~Boundary();

    void printConnectivity() const;
};

#endif

// version
// $Id: Boundary.h,v 1.1 2003/09/09 02:35:22 tan2 Exp $

// End of file
