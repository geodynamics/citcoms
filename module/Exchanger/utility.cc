// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "journal/journal.h"
#include "utility.h"


MPI_Status util::waitRequest(const MPI_Request& request)
{
    MPI_Status status;
    int result = MPI_Wait(const_cast<MPI_Request*>(&request), &status);
    testResult(result, "wait error!");

    return status;
}


std::vector<MPI_Status>
util::waitRequest(const std::vector<MPI_Request>& request)
{
    std::vector<MPI_Status> status(request.size());
    int result = MPI_Waitall(request.size(),
			     const_cast<MPI_Request*>(&request[0]), &status[0]);
    testResult(result, "wait error!");

    return status;
}


void util::testResult(int result, const std::string& errmsg)
{
    if (result != MPI_SUCCESS) {
        journal::error_t error("utility");
        error << journal::loc(__HERE__)
              << errmsg << journal::end;
	throw result;
    }
}





// version
// $Id: utility.cc,v 1.1 2003/11/10 21:55:28 tan2 Exp $

// End of file
