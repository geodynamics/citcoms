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


char pyFull_module__doc__[] = "";

// Initialization function for the module (*must* be called initFull)
extern "C"
void
initFull()
{
    // create the module and add the functions
    PyObject * m = Py_InitModule4(
        "Full", pyCitcom_methods,
        pyFull_module__doc__, 0, PYTHON_API_VERSION);

    // get its dictionary
    PyObject * d = PyModule_GetDict(m);

    // check for errors
    if (PyErr_Occurred()) {
        Py_FatalError("can't initialize module Full");
    }

    // install the module exceptions
    pyCitcom_runtimeError = PyErr_NewException("Full.runtime", 0, 0);
    PyDict_SetItemString(d, "RuntimeException", pyCitcom_runtimeError);

    return;
}

// version
// $Id: Fullmodule.cc,v 1.3 2003/08/01 22:52:05 tan2 Exp $

// End of file
