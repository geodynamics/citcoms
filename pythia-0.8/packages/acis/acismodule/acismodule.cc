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

#include "bindings.h"
#include "exceptions.h"
#include "support.h"


char pyacis_module__doc__[] = "";

// Initialization function for the module (*must* be called initacis)
extern "C"
void
initacis()
{
    // create the module and add the functions
    PyObject * m = Py_InitModule4(
        "acis", pyacis_methods,
        pyacis_module__doc__, 0, PYTHON_API_VERSION);

    // get its dictionary
    PyObject * d = PyModule_GetDict(m);

    // check for errors
    if (PyErr_Occurred()) {
        Py_FatalError("can't initialize module acis");
    }

    // install the module exceptions
    pyacis_runtimeError = PyErr_NewException("acis.runtime", 0, 0);
    PyDict_SetItemString(d, "RuntimeException", pyacis_runtimeError);

    // Start the modeller. This must be the first call to the ACIS library (?)
    if (!ACISModeler::initialize()) {
        Py_FatalError("Can't start ACIS modeller");
        return;
    }

    return;
}

// version
// $Id: acismodule.cc,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $

// End of file
