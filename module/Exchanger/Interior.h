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
    Interior(bool dimensional);
    Interior(const BoundedBox& remoteBBox, const All_variables* E,
	     bool dimensional);
    virtual ~Interior() {};

private:
    void initX(const All_variables* E);

};


#endif

// version
// $Id: Interior.h,v 1.4 2003/12/30 21:46:01 tan2 Exp $

// End of file
