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


char pyelc_module__doc__[] = "";

// Initialization function for the module (*must* be called initelc)
extern "C"
void
initelc()
{
    // create the module and add the functions
    PyObject * m = Py_InitModule4(
        "elc", pyelc_methods,
        pyelc_module__doc__, 0, PYTHON_API_VERSION);

    // get its dictionary
    PyObject * d = PyModule_GetDict(m);

    // check for errors
    if (PyErr_Occurred()) {
        Py_FatalError("can't initialize module elc");
    }

    // install the module exceptions
    pyelc_runtimeError = PyErr_NewException("elc.runtime", 0, 0);
    PyDict_SetItemString(d, "RuntimeException", pyelc_runtimeError);

    return;
}

// version
// $Id: elcmodule.cc,v 1.1.1.1 2005/03/08 16:13:29 aivazis Exp $

// End of file
