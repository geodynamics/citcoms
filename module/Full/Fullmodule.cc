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


char pyFull_module__doc__[] = "";

// Initialization function for the module (*must* be called initFull)
extern "C"
void
initFull()
{
    // create the module and add the functions
    PyObject * m = Py_InitModule4(
        "Full", pyFull_methods,
        pyFull_module__doc__, 0, PYTHON_API_VERSION);

    // get its dictionary
    PyObject * d = PyModule_GetDict(m);

    // check for errors
    if (PyErr_Occurred()) {
        Py_FatalError("can't initialize module Full");
    }

    // install the module exceptions
    pyFull_runtimeError = PyErr_NewException("Full.runtime", 0, 0);
    PyDict_SetItemString(d, "RuntimeException", pyFull_runtimeError);

    return;
}

// version
// $Id: Fullmodule.cc,v 1.2 2003/04/10 23:25:29 tan2 Exp $

// End of file
