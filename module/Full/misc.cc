// -*- C++ -*-
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2003 All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 

#include <portinfo>
#include <Python.h>

#include "misc.h"

extern "C" double return1_test();

// copyright

char pyFull_copyright__doc__[] = "";
char pyFull_copyright__name__[] = "copyright";

static char pyFull_copyright_note[] = 
    "Full python module: Copyright (c) 1998-2003 California Institute of Technology";


PyObject * pyFull_copyright(PyObject *, PyObject *)
{
    return Py_BuildValue("s", pyFull_copyright_note);
}




// hello

char pyFull_return1_test__doc__[] = "";
char pyFull_return1_test__name__[] = "return1_test";

PyObject * pyFull_return1_test(PyObject *, PyObject *)
{
    double a;
    a = return1_test();
    return Py_BuildValue("d", a);
}





// version
// $Id: misc.cc,v 1.2 2003/04/10 23:25:29 tan2 Exp $

// End of file
