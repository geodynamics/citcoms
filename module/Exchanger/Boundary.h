// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#if !defined(pyCitcom_Boundary_h)
#define pyCitcom_Boundary_h

#include <memory>


struct All_variables;


class Boundary {

public:
    static const int dim = 3;  // spatial dimension
    const int size;            // # of boundary nodes

    int *connectivity;
    double *X[dim];            // coordinate

    int *bid2gid;    // bid (local id) -> ID (ie. global id)
    int *bid2proc;   // bid -> proc. rank
    
    explicit Boundary(const int n);     // constructor only allocates memory
    ~Boundary();

    void init(const All_variables *E);  // initialize connectivity and X
    void map(const All_variables *E);   // initialize bid2gid and bid2proc
    void printConnectivity() const;

private:
    std::auto_ptr<int> connectivity_;
    std::auto_ptr<double> X_[dim];

    std::auto_ptr<int> bid2gid_;
    std::auto_ptr<int> bid2proc_;
    
};

#endif

// version
// $Id: Boundary.h,v 1.2 2003/09/09 18:25:31 tan2 Exp $

// End of file
