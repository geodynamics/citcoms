// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#if !defined(pyCitcom_Boundary_h)
#define pyCitcom_Boundary_h

#include "auto_array_ptr.h"
#include "mpi.h"

struct All_variables;


class Boundary {
    const int size_;            // # of boundary nodes

public:
    static const int dim = 3;  // spatial dimension
    auto_array_ptr<double> X[dim];      // coordinate

    double theta_max, theta_min,
	   fi_max, fi_min,
	   ro, ri;   // domain bound of FG

    explicit Boundary(const int n);     // allocating memory only
    ~Boundary() {};

    inline int size() const {return size_;}
    void initBound(const All_variables *E); // init domain bound

    void send(const MPI_Comm comm, const int receiver) const;
    void receive(const MPI_Comm comm, const int sender);
    void broadcast(const MPI_Comm comm, const int broadcaster);

    void printBound() const;
    void printX() const;

private:
    Boundary(const Boundary&);
    Boundary& operator=(const Boundary&);
};

#endif

// version
// $Id: Boundary.h,v 1.17 2003/10/11 00:38:46 tan2 Exp $

// End of file
