// -*- C++ -*-
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2003 All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 

#include <portinfo>

#include <Python.h>

#include "exceptions.h"
#include "bindings.h"


char pyCitcomSFull_module__doc__[] = "";

// Initialization function for the module (*must* be called initCitcomSFull)
extern "C"
void
initCitcomSFull()
{
    // create the module and add the functions
    PyObject * m = Py_InitModule4(
        "CitcomSFull", pyCitcomSFull_methods,
        pyCitcomSFull_module__doc__, 0, PYTHON_API_VERSION);

    // get its dictionary
    PyObject * d = PyModule_GetDict(m);

    // check for errors
    if (PyErr_Occurred()) {
        Py_FatalError("can't initialize module CitcomSFull");
    }

    // install the module exceptions
    pyCitcomSFull_runtimeError = PyErr_NewException("CitcomSFull.runtime", 0, 0);
    PyDict_SetItemString(d, "RuntimeException", pyCitcomSFull_runtimeError);

    return;
}

// version
// $Id: Fullmodule.cc,v 1.1 2003/03/24 01:46:37 tan2 Exp $

// End of file
