// -*- C++ -*-
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2005  All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 

#include <portinfo>

#include <Python.h>

#include "exceptions.h"
#include "bindings.h"


char pypulse_module__doc__[] = "";

// Initialization function for the module (*must* be called initpulse)
extern "C"
void
initpulse()
{
    // create the module and add the functions
    PyObject * m = Py_InitModule4(
        "pulse", pypulse_methods,
        pypulse_module__doc__, 0, PYTHON_API_VERSION);

    // get its dictionary
    PyObject * d = PyModule_GetDict(m);

    // check for errors
    if (PyErr_Occurred()) {
        Py_FatalError("can't initialize module pulse");
    }

    // install the module exceptions
    pypulse_runtimeError = PyErr_NewException("pulse.runtime", 0, 0);
    PyDict_SetItemString(d, "RuntimeException", pypulse_runtimeError);

    return;
}

// version
// $Id: pulsemodule.cc,v 1.1.1.1 2005/03/08 16:13:57 aivazis Exp $

// End of file
