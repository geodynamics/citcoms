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

#include <Python.h>

#include "exceptions.h"
#include "bindings.h"


char pytabulator_module__doc__[] = "";

// Initialization function for the module (*must* be called inittabulator)
extern "C"
void
init_tabulator()
{
    // create the module and add the functions
    PyObject * m = Py_InitModule4(
        "_tabulator", pytabulator_methods,
        pytabulator_module__doc__, 0, PYTHON_API_VERSION);

    // get its dictionary
    PyObject * d = PyModule_GetDict(m);

    // check for errors
    if (PyErr_Occurred()) {
        Py_FatalError("can't initialize module tabulator");
    }

    // install the module exceptions
    pytabulator_runtimeError = PyErr_NewException("_tabulator.runtime", 0, 0);
    PyDict_SetItemString(d, "RuntimeException", pytabulator_runtimeError);

    return;
}

// version
// $Id: _tabulatormodule.cc,v 1.1.1.1 2005/03/17 20:03:02 aivazis Exp $

// End of file
