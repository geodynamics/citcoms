// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_VTInlet_h)
#define pyCitcom_VTInlet_h

#include <string>
#include "Array2D.h"
#include "DIM.h"
#include "Inlet.h"


class VTInlet : public Inlet{
protected:
    const bool modeV;
    const bool modeT;
    const bool modet;
    Array2D<double,DIM> v;
    Array2D<double,DIM> v_old;
    Array2D<double,1> t;
    Array2D<double,1> t_old;

public:
    VTInlet(const BoundedMesh& boundedMesh,
	    const Sink& sink,
	    All_variables* E,
	    const std::string& mode="VT");
    // available mode:
    //     V: impose velocity as BC
    //     T: impose temperature as BC
    //     t: impose temperature but BC is not changed
    //
    // mode "T" and mode "t" cannot co-exist
    //

    virtual ~VTInlet();

    virtual void recv();
    virtual void impose();

private:
    void setVBCFlag();
    void setTBCFlag();
    void recvVT();
    void recvV();
    void recvT();
    void imposeV();
    void imposeT();
    void imposet();


};


#endif

// version
// $Id: VTInlet.h,v 1.2 2004/03/11 01:06:14 tan2 Exp $

// End of file
