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

#include "bindings.h"

#include "misc.h"          // miscellaneous methods

// the method table

struct PyMethodDef pyExchanger_methods[] = {

    // dummy entry for testing
    {pyExchanger_hello__name__, pyExchanger_hello,
     METH_VARARGS, pyExchanger_hello__doc__},

    {pyExchanger_copyright__name__, pyExchanger_copyright,
     METH_VARARGS, pyExchanger_copyright__doc__},


// Sentinel
    {0, 0}
};

// version
// $Id: bindings.cc,v 1.1 2003/09/06 23:44:22 tan2 Exp $

// End of file
