// -*- C++ -*-
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//<LicenseText>
//
// CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
// Copyright (C) 2002-2005, California Institute of Technology.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//</LicenseText>
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 

#include <portinfo>
#include <Python.h>

#include "CitcomSmodule.h"
#include "exceptions.h"
#include "bindings.h"


char pyCitcomS_module__doc__[] = "";

// Initialization function for the module (*must* be called initCitcomSLib)
extern "C"
void
initCitcomSLib()
{
    // create the module and add the functions
    PyObject * m = Py_InitModule4(
        "CitcomSLib", pyCitcom_methods,
        pyCitcomS_module__doc__, 0, PYTHON_API_VERSION);

    // get its dictionary
    PyObject * d = PyModule_GetDict(m);

    // check for errors
    if (PyErr_Occurred()) {
        Py_FatalError("can't initialize module CitcomSLib");
    }

    // install the module exceptions
    pyCitcom_runtimeError = PyErr_NewException("CitcomSLib.runtime", 0, 0);
    PyDict_SetItemString(d, "RuntimeException", pyCitcom_runtimeError);

    return;
}

// version
// $Id$

// End of file
