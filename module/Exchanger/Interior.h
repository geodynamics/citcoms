// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_Interior_h)
#define pyCitcom_Interior_h

#include "BoundedMesh.h"

struct All_variables;


class Interior : public BoundedMesh {

public:
    Interior();
    Interior(const BoundedBox& remoteBBox, const All_variables* E);
    virtual ~Interior() {};

private:
    void initX(const All_variables* E);

};


#endif

// version
// $Id: Interior.h,v 1.6 2004/01/07 21:54:00 tan2 Exp $

// End of file
