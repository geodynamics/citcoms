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

struct PyMethodDef pyCitcomSRegional_methods[] = {

    // dummy entry for testing
    {pyCitcomSRegional_return1_test__name__, pyCitcomSRegional_return1_test,
     METH_VARARGS, pyCitcomSRegional_return1_test__doc__},

    {pyCitcomSRegional_Citcom_Init__name__, pyCitcomSRegional_Citcom_Init,
     METH_VARARGS, pyCitcomSRegional_Citcom_Init__doc__},

    {pyCitcomSRegional_copyright__name__, pyCitcomSRegional_copyright,
     METH_VARARGS, pyCitcomSRegional_copyright__doc__},


// Sentinel
    {0, 0}
};

// version
// $Id: bindings.cc,v 1.2 2003/04/04 00:42:50 tan2 Exp $

// End of file
