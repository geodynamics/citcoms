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
    Interior(bool dimensional, bool transformational);
    Interior(const BoundedBox& remoteBBox, const All_variables* E,
	     bool dimensional, bool transformational);
    virtual ~Interior() {};

private:
    void initX(const All_variables* E);

};


#endif

// version
// $Id: Interior.h,v 1.5 2004/01/06 22:40:28 puru Exp $

// End of file
