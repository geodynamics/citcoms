// -*- C++ -*-
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2005 All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 

#include <portinfo>
#include <Python.h>

#include "bindings.h"

#include "tabulator.h"     // the actual routines
#include "misc.h"          // miscellaneous methods

// the method table

struct PyMethodDef pytabulator_methods[] = {

// misc
    {pytabulator_tabulate__name__, pytabulator_tabulate,
     METH_VARARGS, pytabulator_tabulate__doc__},

    {pytabulator_simpletab__name__, pytabulator_simpletab,
     METH_VARARGS, pytabulator_simpletab__doc__},

    {pytabulator_exponential__name__, pytabulator_exponential,
     METH_VARARGS, pytabulator_exponential__doc__},

    {pytabulator_exponentialSet__name__, pytabulator_exponentialSet,
     METH_VARARGS, pytabulator_exponentialSet__doc__},

    {pytabulator_quadratic__name__, pytabulator_quadratic,
     METH_VARARGS, pytabulator_quadratic__doc__},

    {pytabulator_quadraticSet__name__, pytabulator_quadraticSet,
     METH_VARARGS, pytabulator_quadraticSet__doc__},

    {pytabulator_copyright__name__, pytabulator_copyright,
     METH_VARARGS, pytabulator_copyright__doc__},


// Sentinel
    {0, 0}
};

// version
// $Id: bindings.cc,v 1.1.1.1 2005/03/17 20:03:02 aivazis Exp $

// End of file
