// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcomSExchanger_BoundaryVTInlet_h)
#define pyCitcomSExchanger_BoundaryVTInlet_h

#include "mpi.h"
#include "VTInlet.h"

struct All_variables;
class AreaWeightedNormal;


class BoundaryVTInlet : public VTInlet{
private:
    MPI_Comm comm;
    AreaWeightedNormal* awnormal;

public:
    BoundaryVTInlet(const Boundary& boundary,
		    const Exchanger::Sink& sink,
		    All_variables* E,
		    MPI_Comm comm);
    virtual ~BoundaryVTInlet();

    virtual void recv();

};


#endif

// version
// $Id: BoundaryVTInlet.h,v 1.2 2004/05/11 18:35:24 tan2 Exp $

// End of file
