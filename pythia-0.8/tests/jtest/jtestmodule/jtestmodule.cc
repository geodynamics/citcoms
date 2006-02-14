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


char pyjtest_module__doc__[] = "";

// Initialization function for the module (*must* be called initjtest)
extern "C"
void
initjtest()
{
    // create the module and add the functions
    PyObject * m = Py_InitModule4(
        "jtest", pyjtest_methods,
        pyjtest_module__doc__, 0, PYTHON_API_VERSION);

    // get its dictionary
    PyObject * d = PyModule_GetDict(m);

    // check for errors
    if (PyErr_Occurred()) {
        Py_FatalError("can't initialize module jtest");
    }

    // install the module exceptions
    pyjtest_runtimeError = PyErr_NewException("jtest.runtime", 0, 0);
    PyDict_SetItemString(d, "RuntimeException", pyjtest_runtimeError);

    return;
}

// $Id: jtestmodule.cc,v 1.1.1.1 2005/03/18 17:01:41 aivazis Exp $

// End of file
