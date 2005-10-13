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


char pyremodule_module__doc__[] = "";

// Initialization function for the module (*must* be called init_pyre)
extern "C"
void
init_pyre()
{
    // create the module and add the functions
    PyObject * m = Py_InitModule4(
        "_pyre", pyremodule_methods,
        pyremodule_module__doc__, 0, PYTHON_API_VERSION);

    // get its dictionary
    PyObject * d = PyModule_GetDict(m);

    // check for errors
    if (PyErr_Occurred()) {
        Py_FatalError("can't initialize module _pyre");
    }

    // install the module exceptions
    pyremodule_runtimeError = PyErr_NewException("_pyre.runtime", 0, 0);
    PyDict_SetItemString(d, "RuntimeException", pyremodule_runtimeError);

    return;
}

// version
// $Id: _pyremodule.cc,v 1.1.1.1 2005/03/08 16:13:52 aivazis Exp $

// End of file
