// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_TractionInlet_h)
#define pyCitcom_TractionInlet_h

#include <string>
#include "Array2D.h"
#include "DIM.h"
#include "Inlet.h"

struct All_variables;
class Boundary;
class Sink;


class TractionInlet : public Inlet {
    const bool modeF;
    const bool modeV;
    Array2D<double,DIM> f;
    Array2D<double,DIM> f_old;
    Array2D<double,DIM> v;
    Array2D<double,DIM> v_old;

public:
    TractionInlet(const Boundary& boundary,
		  const Sink& sink,
		  All_variables* E,
		  const std::string& mode="F");
    virtual ~TractionInlet();

    virtual void recv();
    virtual void impose();

private:
    void setVBCFlag();
    void setMixedBC();
    void setVBC();
    void setFBC();

    void recvFV();
    void recvF();
    void recvV();

    void imposeFV();
    void imposeF();
    void imposeV();

};


#endif

// version
// $Id: TractionInlet.h,v 1.1 2004/03/28 23:05:19 tan2 Exp $

// End of file
