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

};


#endif

// version
// $Id: Interior.h,v 1.2 2003/11/10 21:55:28 tan2 Exp $

// End of file
