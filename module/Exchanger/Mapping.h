// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#if !defined(pyCitcom_Mapping_h)
#define pyCitcom_Mapping_h

#include "auto_array_ptr.h"
#include "mpi.h"

struct All_variables;
class Boundary;


class Mapping {
protected:
    int size_;
    const int memsize_;
    auto_array_ptr<int> bid2proc_; // bid -> proc. rank

public:
    explicit Mapping(const int size);
    virtual ~Mapping() = 0;

    inline const int size() {return size_;}
    inline int bid2proc(const int n) const {return bid2proc_[n];};

protected:
    void sendBid2proc(const MPI_Comm comm,
		      const int rank, const int leader);
                                         // send bid2proc to leader
    void printBid2proc() const;

};



class CoarseGridMapping : public Mapping {
    auto_array_ptr<int> bid2elem_; // bid -> elem from which fields are interpolated in CG
    auto_array_ptr<double> shape_; // shape functions for interpolation

public:
    CoarseGridMapping(const Boundary* b, const All_variables* E,
		      const MPI_Comm comm,
		      const int rank, const int leader);
    virtual ~CoarseGridMapping() {};

    inline int bid2elem(const int n) const {return bid2elem_[n];};
    inline double shape(const int n) const {return shape_[n];};

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

    void printBid2elem() const;

};



class FineGridMapping : public Mapping {
    auto_array_ptr<int> bid2gid_;  // bid (local id) -> ID (ie. global id in FG)

public:
    explicit FineGridMapping(Boundary* b, const All_variables* E,
			     const MPI_Comm comm,
			     const int rank, const int leader);
    virtual ~FineGridMapping() {};

    inline int bid2gid(const int n) const {return bid2gid_[n];};

private:
    void findBoundaryNodes(Boundary* boundary, const All_variables* E);
    void printBid2gid() const;

};


#endif

// version
// $Id: Mapping.h,v 1.1 2003/10/11 00:38:46 tan2 Exp $

// End of file
