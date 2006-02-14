// -*- C++ -*-
//
//-----------------------------------------------------------------------------
//
//                              Michael A.G. Aivazis
//                       California Institute of Technology
//                       (C) 1998-2005  All Rights Reserved
//
// <LicenseText>
//
//-----------------------------------------------------------------------------
//

#include <portinfo>
#include <Python.h>
#include <mpi.h>

#include <cmath>

#include "driver.h"
#include "journal/debug.h"

char pypulse_timestep__doc__[] = "compute stable timestep";
char pypulse_timestep__name__[] = "timestep";
PyObject * pypulse_timestep(PyObject *, PyObject * args)
{
    journal::debug_t info("pulse");
    double dtf;
    int fluidServer;
    int solidServer;

    int ok = PyArg_ParseTuple(args, "iid:timestep", &fluidServer, &solidServer,
        &dtf);

    if (!ok) {
        return 0;
    }

    double timestep = dtf;

#if defined(PARALLEL) && defined(WITH_MPI)
    MPI_Status status;
    info << journal::at(__HERE__) << "fluid server is world rank "
         << fluidServer << journal::endl;
    info << journal::at(__HERE__) << "solid server is world rank "
         << solidServer << journal::endl;

    // exchange with solid server and select stable timestep
    double dts;
    const int ftag = 18;
    info << journal::at(__HERE__) 
         << "sending fluid timestep proposal to the solid server"
         << journal::endl;
    MPI_Send(&dtf, 1, MPI_DOUBLE, solidServer, ftag, MPI_COMM_WORLD);
    const int stag = 19;
    info << journal::at(__HERE__) 
         << "receiving solid timestep proposal from the solid server"
         << journal::endl;
    MPI_Recv(&dts, 1, MPI_DOUBLE, solidServer, stag, MPI_COMM_WORLD, 
        &status);
    timestep = (dtf<dts) ? dtf : dts;
#endif

    // return the value
    return PyFloat_FromDouble(timestep);
}


// $Id: driver.cc,v 1.1.1.1 2005/03/08 16:13:57 aivazis Exp $

// End of file
