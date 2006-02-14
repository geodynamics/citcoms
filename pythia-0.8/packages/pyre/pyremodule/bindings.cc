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

#include "geometry.h"      // geometry support
#include "misc.h"          // miscellaneous methods

// the method table

struct PyMethodDef pyremodule_methods[] = {

    // geometry
    {pyremodule_createMesh__name__, pyremodule_createMesh,
     METH_VARARGS, pyremodule_createMesh__doc__},

    {pyremodule_statistics__name__, pyremodule_statistics,
     METH_VARARGS, pyremodule_statistics__doc__},

    {pyremodule_vertex__name__, pyremodule_vertex, METH_VARARGS, pyremodule_vertex__doc__},
    {pyremodule_simplex__name__, pyremodule_simplex, METH_VARARGS, pyremodule_simplex__doc__},

    // copyright note
    {pyremodule_copyright__name__, pyremodule_copyright,
     METH_VARARGS, pyremodule_copyright__doc__},


// Sentinel
    {0, 0}
};

// version
// $Id: bindings.cc,v 1.1.1.1 2005/03/08 16:13:52 aivazis Exp $

// End of file
