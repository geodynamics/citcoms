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

#include "bindings.h"

#include "misc.h"          // miscellaneous methods

// the method table

struct PyMethodDef pyRegional_methods[] = {

    // dummy entry for testing
    {pyRegional_copyright__name__, pyRegional_copyright,
     METH_VARARGS, pyRegional_copyright__doc__},

    {pyRegional_return1_test__name__, pyRegional_return1_test,
     METH_VARARGS, pyRegional_return1_test__doc__},

    {pyRegional_Citcom_Init__name__, pyRegional_Citcom_Init,
     METH_VARARGS, pyRegional_Citcom_Init__doc__},

    {pyRegional_read_instructions__name__, pyRegional_read_instructions,
     METH_VARARGS, pyRegional_read_instructions__doc__},



// Sentinel
    {0, 0}
};

// version
// $Id: bindings.cc,v 1.4 2003/04/10 23:18:24 tan2 Exp $

// End of file
