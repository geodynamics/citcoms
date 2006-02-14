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

#include "bindings.h"

#include "misc.h"          // miscellaneous methods

// the method table

struct PyMethodDef pyjtest_methods[] = {

    // 
    {pyjtest_info__name__, pyjtest_info, METH_VARARGS, pyjtest_info__doc__},
    {pyjtest_error__name__, pyjtest_error, METH_VARARGS, pyjtest_error__doc__},
    {pyjtest_warning__name__, pyjtest_warning, METH_VARARGS, pyjtest_warning__doc__},

    {pyjtest_copyright__name__, pyjtest_copyright,
     METH_VARARGS, pyjtest_copyright__doc__},


// Sentinel
    {0, 0}
};

// $Id: bindings.cc,v 1.1.1.1 2005/03/18 17:01:41 aivazis Exp $

// End of file
