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
#include <cstdio>
#include <iostream>

#include "outputs.h"

extern "C" {
#include "global_defs.h"
#include "output.h"

}


char pyCitcom_output__doc__[] = "";
char pyCitcom_output__name__[] = "output";

PyObject * pyCitcom_output(PyObject *self, PyObject *args)
{
    PyObject *obj;
    int cycles;

    if (!PyArg_ParseTuple(args, "Oi:output", &obj, &cycles))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    output(E, cycles);

    Py_INCREF(Py_None);
    return Py_None;
}




// version
// $Id: outputs.cc,v 1.10 2003/08/19 21:21:43 tan2 Exp $

// End of file
