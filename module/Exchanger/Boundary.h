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
    const double theta_max, theta_min, phi_max, phi_min, rout, rin;
    
    int *connectivity;
    double *X[dim];            // coordinate

    int *bid2gid;    // bid (local id) -> ID (ie. global id)
    int *bid2proc;   // bid -> proc. rank

    explicit Boundary(const int n);     // constructor only allocates memory
    ~Boundary();

    void init(const All_variables *E);  // initialize connectivity and X
    void mapFineGrid(const All_variables *E, int localLeader);
    void mapCoarseGrid(const All_variables *E, int localLeader);
                                        // initialize bid2gid and bid2proc
    void printConnectivity() const;
    void printX() const;
    void printBid2gid() const;

private:
//     Boundary(const int,
// 	     std::auto_ptr<int>,
// 	     std::auto_ptr<double>,
// 	     std::auto_ptr<double>,
// 	     std::auto_ptr<double>);

//     const std::auto_ptr<int> connectivity_;
//     const std::auto_ptr<double> X_[dim];

//     std::auto_ptr<int> bid2gid_;
//     std::auto_ptr<int> bid2proc_;

};

#endif

// version
// $Id: Boundary.h,v 1.6 2003/09/18 17:16:44 puru Exp $

// End of file
