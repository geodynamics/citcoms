// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_AreaWeightedNormal_h)
#define pyCitcom_AreaWeightedNormal_h

#include <vector>
#include "DIM.h"

struct All_variables;
template <class T, int N> class Array2D;
class Boundary;
class Sink;


class AreaWeightedNormal {
    const int size_;
    const double toleranceOutflow_;
    double total_area_;
    std::vector<double> nwght;

public:
    AreaWeightedNormal(const MPI_Comm& comm,
		       const Boundary& boundary,
		       const Sink& sink,
		       const All_variables* E);
    ~AreaWeightedNormal() {};

    typedef Array2D<double,DIM> Velo;

    void imposeConstraint(Velo& V,
			  const MPI_Comm& comm,
			  const Sink& sink) const;

private:
    void computeWeightedNormal(const Boundary& boundary,
			       const Sink& sink,
			       const All_variables* E);
    void computeTotalArea(const MPI_Comm& comm, const Sink& sink);
    double computeOutflow(const Velo& V,
			  const MPI_Comm& comm,
			  const Sink& sink) const;
    void reduceOutflow(Velo& V, double outflow, const Sink& sink) const;

};


#endif

// version
// $Id: AreaWeightedNormal.h,v 1.3 2003/11/10 21:55:28 tan2 Exp $

// End of file

