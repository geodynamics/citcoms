// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcomSExchanger_SIUnit_h)
#define pyCitcomSExchanger_SIUnit_h

#include "Exchanger/SIUnit.h"

struct All_variables;


// singleton class

class SIUnit : public Exchanger::SIUnit {

public:
    SIUnit(const All_variables* E);
    virtual ~SIUnit();

private:
    // disable
    SIUnit(const SIUnit&);
    SIUnit& operator=(const SIUnit&);

};


#endif

// version
// $Id: SIUnit.h,v 1.4 2004/05/11 07:55:30 tan2 Exp $

// End of file
