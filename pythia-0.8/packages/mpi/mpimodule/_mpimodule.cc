// -*- C++ -*-
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2005 All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 

#include <portinfo>

#include <mpi.h>
#include <Python.h>

#include "exceptions.h"
#include "startup.h"
#include "bindings.h"

#include "Communicator.h"

char pympi_module__doc__[] = "";

// Initialization function for the module (*must* be called init_mpi)
extern "C"
void
init_mpi()
{
    // create the module and add the functions
    PyObject * m = Py_InitModule4(
        "_mpi", pympi_methods,
        pympi_module__doc__, 0, PYTHON_API_VERSION);

    // get its dictionary
    PyObject * d = PyModule_GetDict(m);

    // check for errors
    if (PyErr_Occurred()) {
        Py_FatalError("can't initialize module _mpi");
    }

    // install the module exceptions
    pympi_runtimeError = PyErr_NewException("mpi.runtime", 0, 0);
    PyDict_SetItemString(d, "RuntimeException", pympi_runtimeError);

// initialize MPI
    int ok = pympi_initialize();

// add some constants
    PyDict_SetItemString(d, "initialized", PyLong_FromLong(ok));
    PyDict_SetItemString(
        d, "world", PyCObject_FromVoidPtr(new mpi::Communicator(MPI_COMM_WORLD), 0));

    return;
}

// version
// $Id: _mpimodule.cc,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $

// End of file
