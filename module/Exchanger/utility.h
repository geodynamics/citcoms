// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_utility_h)
#define pyCitcom_utility_h

#include <vector>
#include "mpi.h"

namespace util {

    MPI_Status waitRequest(const MPI_Request& request);

    std::vector<MPI_Status>
    waitRequest(const std::vector<MPI_Request>& request);

    void testResult(int result, const std::string& errmsg);

}

#endif

// version
// $Id: utility.h,v 1.1 2003/11/10 21:55:28 tan2 Exp $

// End of file
