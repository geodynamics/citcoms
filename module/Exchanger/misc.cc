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

extern "C" {
#include "global_defs.h"
}

#include "misc.h"


// copyright

char pyExchanger_copyright__doc__[] = "";
char pyExchanger_copyright__name__[] = "copyright";

static char pyExchanger_copyright_note[] =
    "Exchanger python module: Copyright (c) 1998-2003 California Institute of Technology";


PyObject * pyExchanger_copyright(PyObject *, PyObject *)
{
    return Py_BuildValue("s", pyExchanger_copyright_note);
}

// hello

char pyExchanger_hello__doc__[] = "";
char pyExchanger_hello__name__[] = "hello";

PyObject * pyExchanger_hello(PyObject *, PyObject *)
{
    return Py_BuildValue("s", "hello");
}

// return (All_variables* E)

char pyExchanger_returnE__doc__[] = "";
char pyExchanger_returnE__name__[] = "returnE";

PyObject * pyExchanger_returnE(PyObject *, PyObject *)
{
    All_variables *E = new All_variables;

    E->parallel.me = 1;

    PyObject *cobj = PyCObject_FromVoidPtr(E, NULL);
    return Py_BuildValue("O", cobj);
}

// version
// $Id: misc.cc,v 1.3 2003/09/09 20:57:25 tan2 Exp $

// End of file
