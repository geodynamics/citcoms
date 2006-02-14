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
#include "misc.h"          // miscellaneous methods

// the method table

struct PyMethodDef pyrigid_methods[] = {

    // driver
    {pyrigid_timestep__name__, pyrigid_timestep, METH_VARARGS,
     pyrigid_timestep__doc__},

     // copyright note
    {pyrigid_copyright__name__, pyrigid_copyright,
     METH_VARARGS, pyrigid_copyright__doc__},


// Sentinel
    {0, 0}
};

// version
// $Id: bindings.cc,v 1.1.1.1 2005/03/08 16:13:58 aivazis Exp $

// End of file
