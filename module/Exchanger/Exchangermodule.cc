// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>

#include <Python.h>

#include "exceptions.h"
#include "bindings.h"


char pyExchanger_module__doc__[] = "";

// Initialization function for the module (*must* be called initExchanger)
extern "C"
void
initExchanger()
{
    // create the module and add the functions
    PyObject * m = Py_InitModule4(
        "Exchanger", pyExchanger_methods,
        pyExchanger_module__doc__, 0, PYTHON_API_VERSION);

    // get its dictionary
    PyObject * d = PyModule_GetDict(m);

    // check for errors
    if (PyErr_Occurred()) {
        Py_FatalError("can't initialize module Exchanger");
    }

    // install the module exceptions
    pyExchanger_runtimeError = PyErr_NewException("Exchanger.runtime", 0, 0);
    PyDict_SetItemString(d, "RuntimeException", pyExchanger_runtimeError);

    return;
}

// version
// $Id: Exchangermodule.cc,v 1.2 2003/11/07 01:08:01 tan2 Exp $

// End of file
