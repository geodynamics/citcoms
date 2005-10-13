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

#include "misc.h"
#include "journal/diagnostics.h"

// copyright
char pyjtest_copyright__doc__[] = "";
char pyjtest_copyright__name__[] = "copyright";

static char pyjtest_copyright_note[] = 
    "jtest python module: Copyright (c) 1998-2005 Michael A.G. Aivazis";


PyObject * pyjtest_copyright(PyObject *, PyObject *)
{
    return Py_BuildValue("s", pyjtest_copyright_note);
}
    
// info
char pyjtest_info__doc__[] = "";
char pyjtest_info__name__[] = "info";

PyObject * pyjtest_info(PyObject *, PyObject * args)
{
    char * category;
    int ok = PyArg_ParseTuple(args, "s:info", &category); 

    if (!ok) {
        return 0;
    }

    journal::info_t info(category);
    journal::info_t info2(category);

    info
        << journal::at(__HERE__)
        << "this is the first line of the message" << journal::newline
        << "this is the second line of the message" << journal::endl;

    // return
    Py_INCREF(Py_None);
    return Py_None;
}
    
// error
char pyjtest_error__doc__[] = "";
char pyjtest_error__name__[] = "error";

PyObject * pyjtest_error(PyObject *, PyObject * args)
{
    char * category;
    int ok = PyArg_ParseTuple(args, "s:error", &category); 

    if (!ok) {
        return 0;
    }

    journal::error_t error(category);

    error
        << journal::at(__HERE__)
        << "this is the first line of the message" << journal::newline
        << "this is the second line of the message" << journal::endl;

    // return
    Py_INCREF(Py_None);
    return Py_None;
}
    
// warning
char pyjtest_warning__doc__[] = "";
char pyjtest_warning__name__[] = "warning";

PyObject * pyjtest_warning(PyObject *, PyObject * args)
{
    char * category;
    int ok = PyArg_ParseTuple(args, "s:warning", &category); 

    if (!ok) {
        return 0;
    }

    journal::warning_t warning(category);

    warning
        << journal::at(__HERE__)
        << "this is the first line of the message" << journal::newline
        << "this is the second line of the message" << journal::endl;

    // return
    Py_INCREF(Py_None);
    return Py_None;
}
    
// $Id: misc.cc,v 1.1.1.1 2005/03/18 17:01:42 aivazis Exp $

// End of file
