// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcomSExchanger_Interior_h)
#define pyCitcomSExchanger_Interior_h

#include "Exchanger/BoundedMesh.h"

struct All_variables;


class Interior : public Exchanger::BoundedMesh {

public:
    Interior();
    Interior(const Exchanger::BoundedBox& remoteBBox,
	     const All_variables* E);
    virtual ~Interior();

private:
    void initX(const All_variables* E);

};


#endif

// version
// $Id: Interior.h,v 1.7 2004/05/11 07:55:30 tan2 Exp $

// End of file
