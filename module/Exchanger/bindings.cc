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

#include "exchangers.h"
#include "misc.h"          // miscellaneous methods

// the method table

struct PyMethodDef pyExchanger_methods[] = {

    // dummy entry for testing
    {pyExchanger_hello__name__,
     pyExchanger_hello,
     METH_VARARGS,
     pyExchanger_hello__doc__},

    {pyExchanger_copyright__name__,
     pyExchanger_copyright,
     METH_VARARGS,
     pyExchanger_copyright__doc__},

    // from exchangers.h

    {pyExchanger_returnE__name__,
     pyExchanger_returnE,
     METH_VARARGS,
     pyExchanger_returnE__doc__},

    {pyExchanger_createCoarseGridExchanger__name__,
     pyExchanger_createCoarseGridExchanger,
     METH_VARARGS,
     pyExchanger_createCoarseGridExchanger__doc__},

    {pyExchanger_createFineGridExchanger__name__,
     pyExchanger_createFineGridExchanger,
     METH_VARARGS,
     pyExchanger_createFineGridExchanger__doc__},

    {pyExchanger_createBoundary__name__,
     pyExchanger_createBoundary,
     METH_VARARGS,
     pyExchanger_createBoundary__doc__},

    {pyExchanger_receiveBoundary__name__,
     pyExchanger_receiveBoundary,
     METH_VARARGS,
     pyExchanger_receiveBoundary__doc__},

    {pyExchanger_sendBoundary__name__,
     pyExchanger_sendBoundary,
     METH_VARARGS,
     pyExchanger_sendBoundary__doc__},




// Sentinel
    {0, 0}
};

// version
// $Id: bindings.cc,v 1.3 2003/09/09 02:35:22 tan2 Exp $

// End of file
