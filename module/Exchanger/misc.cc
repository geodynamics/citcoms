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

#include "misc.h"
#include "src/hello.h"


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
    return Py_BuildValue("s", hello());
}

// version
// $Id: misc.cc,v 1.1 2003/09/06 23:44:22 tan2 Exp $

// End of file
