// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#if !defined(pyCitcom_Boundary_h)
#define pyCitcom_Boundary_h

#include <string>
#include "mpi.h"
#include "Array2D.h"

struct All_variables;


class Boundary {
    static const int dim_ = 3;      // spatial dimension
    int size_;                      // # of boundary nodes
    Array2D<double, dim_> bounds_;  // domain bounds
    Array2D<double, dim_> X_;       // coordinate

public:
    explicit Boundary(const All_variables* E);  // allocate memory and init domain bounds
    explicit Boundary(const int n);     // allocate memory only
    explicit Boundary(const All_variables* E, Boundary *b);  // allocate memory and set the interior nodes
    
    Boundary();
    ~Boundary() {};

    inline int dim() const {return dim_;}
    inline int size() const {return size_;}
    inline double theta_max() const {return bounds_[0][0];}
    inline double theta_min() const {return bounds_[0][1];}
    inline double fi_max() const {return bounds_[1][0];}
    inline double fi_min() const {return bounds_[1][1];}
    inline double ro() const {return bounds_[2][0];}
    inline double ri() const {return bounds_[2][1];}

    inline double X(const int d, const int n) const {return X_[d][n];}
    inline void setX(const int d, const int n, const double val) {X_[d][n] = val;}

    void send(const MPI_Comm comm, const int receiver) const;
    void receive(const MPI_Comm comm, const int sender);
    void broadcast(const MPI_Comm comm, const int broadcaster);

    void resize(const int n);

    void printBounds(const std::string& prefix="") const;
    void printX(const std::string& prefix="") const;

private:
    Boundary(const Boundary&);
    Boundary& operator=(const Boundary&);

    void initBounds(const All_variables *E);
};
#endif

// version
// $Id: Boundary.h,v 1.21 2003/10/28 02:34:37 puru Exp $

// End of file
