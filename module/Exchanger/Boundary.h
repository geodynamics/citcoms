// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcomSExchanger_Boundary_h)
#define pyCitcomSExchanger_Boundary_h

#include "Exchanger/Boundary.h"

struct All_variables;


class Boundary : public Exchanger::Boundary {

public:
    Boundary();
    explicit Boundary(const All_variables* E);
    virtual ~Boundary();

private:
    void initBBox(const All_variables *E);
    void initX(const All_variables *E);

};


#endif

// version
// $Id: Boundary.h,v 1.30 2004/05/11 07:55:30 tan2 Exp $

// End of file
