// -*- C++ -*-
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//  <LicenseText>
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        "Regional", pyCitcom_methods,
        pyRegional_module__doc__, 0, PYTHON_API_VERSION);

    // get its dictionary
    PyObject * d = PyModule_GetDict(m);

    // check for errors
    if (PyErr_Occurred()) {
        Py_FatalError("can't initialize module Regional");
    }

    // install the module exceptions
    pyCitcom_runtimeError = PyErr_NewException("Regional.runtime", 0, 0);
    PyDict_SetItemString(d, "RuntimeException", pyCitcom_runtimeError);

    return;
}

// version
// $Id: Regionalmodule.cc,v 1.3 2003/08/01 22:53:50 tan2 Exp $

// End of file
