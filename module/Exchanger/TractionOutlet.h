// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_TractionOutlet_h)
#define pyCitcom_TractionOutlet_h

#include "Array2D.h"
#include "DIM.h"
#include "Outlet.h"

struct All_variables;
class TractionSource;


class TractionOutlet : public Outlet {
    const bool modeF;
    const bool modeV;
    Array2D<double,DIM> f;
    Array2D<double,DIM> v;

public:
    TractionOutlet(TractionSource& src,
		   All_variables* E,
		   const std::string& mode="F");
    virtual ~TractionOutlet();

    virtual void send();

private:
    void sendFV();
    void sendF();
    void sendV();

    // disable copy c'tor and assignment operator
    TractionOutlet(const TractionOutlet&);
    TractionOutlet& operator=(const TractionOutlet&);

};


#endif

// version
// $Id: TractionOutlet.h,v 1.2 2004/03/28 23:19:00 tan2 Exp $

// End of file
