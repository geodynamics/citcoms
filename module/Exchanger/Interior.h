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
    explicit Interior(const BoundedBox& remoteBBox, const All_variables* E);
    virtual ~Interior() {};

    virtual void broadcast(const MPI_Comm& comm, int broadcaster);

};


#endif

// version
// $Id: Interior.h,v 1.1 2003/11/07 01:08:01 tan2 Exp $

// End of file
