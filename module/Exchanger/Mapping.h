// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#if !defined(pyCitcom_Mapping_h)
#define pyCitcom_Mapping_h

#include <string>
#include "mpi.h"
#include "Array2D.h"

struct All_variables;
class Boundary;


class Mapping {
protected:
    const int dim_;
    int size_;
    Array2D<int,1> bid2proc_; // bid -> proc. rank

public:
    Mapping(const int dim, const int size);
    virtual ~Mapping() = 0;

    inline const int size() const {return size_;}
    inline int bid2proc(const int n) const {return bid2proc_[0][n];};
    void printBid2proc(const std::string& prefix="") const;
    virtual void resize(const int n);

protected:
    void sendBid2proc(const MPI_Comm comm,
		      const int rank, const int leader);
                                         // send bid2proc to leader
};



class CoarseGridMapping : public Mapping {
    Array2D<int,1> bid2elem_; // bid -> elem from which fields are interpolated in CG
    Array2D<double,1> shape_; // shape functions for interpolation
    int interiornodes;
    double *Xinterior;
    
public:
    CoarseGridMapping(const Boundary* b, const All_variables* E,
		      const MPI_Comm comm,
		      const int rank, const int leader);
    virtual ~CoarseGridMapping() {};

    inline int bid2elem(const int n) const {return bid2elem_[0][n];};
    inline double shape(const int n) const {return shape_[0][n];};
    void printBid2elem(const std::string& prefix="") const;
    virtual void resize(const int n);
    void FindInteriorNodes(const Boundary* boundary,const All_variables* E);

private:
    void findMaxGridSpacing(const All_variables* E, double& theta_tol,
			    double& fi_tol, double& r_tol) const;

    void findBoundaryElements(const Boundary* boundary, const All_variables* E,
			      const int rank, const double theta_tol,
			      const double  fi_tol, const double  r_tol);

    void selfTest(const Boundary* boundary, const All_variables *E) const;

    double TetrahedronVolume(double *x1, double *x2,
			     double *x3, double *x4) const;

    double det3_sub(double  *x1, double *x2, double *x3) const;
};



class FineGridMapping : public Mapping {
    Array2D<int,1> bid2gid_;  // bid (local id) -> ID (ie. global id in FG)

public:
    explicit FineGridMapping(Boundary* b, const All_variables* E,
			     const MPI_Comm comm,
			     const int rank, const int leader);
    virtual ~FineGridMapping() {};

    inline int bid2gid(const int n) const {return bid2gid_[0][n];};
    void printBid2gid(const std::string& prefix="") const;
    virtual void resize(const int n);

private:
    void findBoundaryNodes(Boundary* boundary, const All_variables* E);
};


#endif

// version
// $Id: Mapping.h,v 1.7 2003/10/23 19:06:24 puru Exp $

// End of file
