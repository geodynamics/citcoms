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

#include "journal/debug.h"

#include "driver.h"

// bindings

char pyrigid_timestep__doc__[] = "compute stable timestep";
char pyrigid_timestep__name__[] = "timestep";
PyObject * pyrigid_timestep(PyObject *, PyObject * args)
{
    journal::debug_t info("rigid");

    int fluidServer;
    int solidServer;
    double dts;

    int ok = PyArg_ParseTuple(args, "iid:timestep", &fluidServer, &solidServer,
        &dts);

    if (!ok) {
        return 0;
    }

    double timestep = dts;

#if defined(PARALLEL) && defined(WITH_MPI)
    MPI_Status status;
    info << journal::at(__HERE__) << "fluid server is world rank "
         << fluidServer << journal::endl;
    info << journal::at(__HERE__) << "solid server is world rank "
         << solidServer << journal::endl;

    // exchange with fluid server and select stable timestep
    double dtf;
    const int ftag = 18;
    info << journal::at(__HERE__)
         << "receiving fluid timestep proposal from the fluid server"
         << journal::endl;
    MPI_Recv(&dtf, 1, MPI_DOUBLE, fluidServer, ftag, MPI_COMM_WORLD, &status);
    const int stag = 19;
    info << journal::at(__HERE__)
         << "sending solid timestep proposal to the fluid server"
         << journal::endl;
    MPI_Send(&dts, 1, MPI_DOUBLE, fluidServer, stag, MPI_COMM_WORLD);

    // select stable timestep and apply desired reduction factor
    timestep = (dtf<dts) ? dtf : dts;
#endif

    // return this value
    return PyFloat_FromDouble(timestep);
}


// $Id: driver.cc,v 1.1.1.1 2005/03/08 16:13:58 aivazis Exp $

// End of file
