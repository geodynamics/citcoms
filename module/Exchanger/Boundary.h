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
    double theta_max, theta_min, fi_max, fi_min, ro, ri;
    
    int *connectivity;
    double *X[dim];            // coordinate

    int *bid2gid;    // bid (local id) -> ID (ie. global id)
    int *bid2proc;   // bid -> proc. rank
  // bid2crseelem provides for the element number in coarse mesh from which to interpolate
    int *bid2crseelem;
    double *shape;
    explicit Boundary(const int n);     // constructor only allocates memory
    ~Boundary();

    void init(const All_variables *E);  // initialize connectivity and X
    void getBid2crseelem(const All_variables *E);
    void mapFineGrid(const All_variables *E, int localLeader);
    void mapCoarseGrid(const All_variables *E, int localLeader);
                                        // initialize bid2gid and bid2proc
    void printConnectivity() const;
    void printX() const;
    void printBid2gid() const;
    double Tetrahedronvolume(double  *x1, double *x2, double *x3, double *x4);
    double det3_sub(double  **xx);
    
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
// $Id: Boundary.h,v 1.10 2003/09/24 04:53:10 puru Exp $

// End of file
