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


char pyRegional_module__doc__[] = "";

// Initialization function for the module (*must* be called initRegional)
extern "C"
void
initRegional()
{
    // create the module and add the functions
    PyObject * m = Py_InitModule4(
        "Regional", pyRegional_methods,
        pyRegional_module__doc__, 0, PYTHON_API_VERSION);

    // get its dictionary
    PyObject * d = PyModule_GetDict(m);

    // check for errors
    if (PyErr_Occurred()) {
        Py_FatalError("can't initialize module Regional");
    }

    // install the module exceptions
    pyRegional_runtimeError = PyErr_NewException("Regional.runtime", 0, 0);
    PyDict_SetItemString(d, "RuntimeException", pyRegional_runtimeError);

    return;
}

// version
// $Id: Regionalmodule.cc,v 1.2 2003/04/10 23:18:24 tan2 Exp $

// End of file
