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
#include "mpi.h"
#include "global_defs.h"
#include "citcom_init.h"
#include "output.h"

}


char pyRegional_output__doc__[] = "";
char pyRegional_output__name__[] = "output";

PyObject * pyRegional_output(PyObject *self, PyObject *args)
{
    int cycles;

    if (!PyArg_ParseTuple(args, "i:output", &cycles))
        return NULL;

    output(E, cycles);

    Py_INCREF(Py_None);
    return Py_None;
}




// version
// $Id: outputs.cc,v 1.8 2003/07/28 21:57:01 tan2 Exp $

// End of file
