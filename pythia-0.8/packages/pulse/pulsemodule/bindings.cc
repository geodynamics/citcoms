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

#include "driver.h"        // timestep negotiation
#include "generators.h"    // pulse generators
#include "misc.h"          // miscellaneous methods

// the method table

struct PyMethodDef pypulse_methods[] = {

    // generators
    {pypulse_heaviside__name__, pypulse_heaviside, METH_VARARGS,
     pypulse_heaviside__doc__},
    {pypulse_bath__name__, pypulse_bath, METH_VARARGS, pypulse_bath__doc__},

    // driver
    {pypulse_timestep__name__, pypulse_timestep, METH_VARARGS,
     pypulse_timestep__doc__},

    // copyright note
    {pypulse_copyright__name__, pypulse_copyright,
     METH_VARARGS, pypulse_copyright__doc__},


// Sentinel
    {0, 0}
};

// version
// $Id: bindings.cc,v 1.1.1.1 2005/03/08 16:13:57 aivazis Exp $

// End of file
