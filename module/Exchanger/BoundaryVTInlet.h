// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_BoundaryVTInlet_h)
#define pyCitcom_BoundaryVTInlet_h

#include <string>
#include "mpi.h"
#include "VTInlet.h"

struct All_variables;
class AreaWeightedNormal;
class Boundary;
class Sink;


class BoundaryVTInlet : public VTInlet{
private:
    MPI_Comm comm;
    AreaWeightedNormal* awnormal;

public:
    BoundaryVTInlet(MPI_Comm comm, const Boundary& boundary,
		    const Sink& sink, All_variables* E,
		    const std::string& mode="VT");
    virtual ~BoundaryVTInlet();

    virtual void recv();

};


#endif

// version
// $Id: BoundaryVTInlet.h,v 1.1 2004/02/24 20:34:43 tan2 Exp $

// End of file
