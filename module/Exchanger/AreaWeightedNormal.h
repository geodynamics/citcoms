// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcomSExchanger_AreaWeightedNormal_h)
#define pyCitcomSExchanger_AreaWeightedNormal_h

#include <vector>
#include "mpi.h"
#include "Exchanger/Array2D.h"
#include "Exchanger/DIM.h"
#include "Exchanger/Sink.h"

struct All_variables;
class Boundary;


class AreaWeightedNormal {
    const int size_;
    const double toleranceOutflow_;
    double total_area_;
    std::vector<double> nwght;

public:
    AreaWeightedNormal(const MPI_Comm& comm,
		       const Boundary& boundary,
		       const Exchanger::Sink& sink,
		       const All_variables* E);
    ~AreaWeightedNormal();

    typedef Exchanger::Array2D<double,Exchanger::DIM> Velo;

    void imposeConstraint(Velo& V,
			  const MPI_Comm& comm,
			  const Exchanger::Sink& sink) const;

private:
    void computeWeightedNormal(const Boundary& boundary,
			       const All_variables* E);
    void computeTotalArea(const MPI_Comm& comm,
			  const Exchanger::Sink& sink);
    double computeOutflow(const Velo& V,
			  const MPI_Comm& comm,
			  const Exchanger::Sink& sink) const;
    void reduceOutflow(Velo& V, double outflow,
		       const Exchanger::Sink& sink) const;

};


#endif

// version
// $Id: AreaWeightedNormal.h,v 1.6 2004/05/11 18:35:24 tan2 Exp $

// End of file

