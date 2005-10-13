// -*- C++ -*-
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2005  All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 

#include <portinfo>
#include <Python.h>

#include "pyre/geometry/CanonicalMesh.h"
typedef pyre::geometry::CanonicalMesh<double> mesh_t;

#include "journal/debug.h"

#include "verify.h"

// verify

char pyelc_verify__doc__[] = "sanity checks on solid boundary";
char pyelc_verify__name__[] = "verify";

PyObject * pyelc_verify(PyObject *, PyObject * args)
{
    PyObject * py_mesh;

    int ok = PyArg_ParseTuple(args, "O:verify", &py_mesh); 

    if (!ok) {
        return 0;
    }

    mesh_t * mesh = static_cast<mesh_t *>(PyCObject_AsVoidPtr(py_mesh));

    journal::debug_t info("elc");
    info 
        << journal::at(__HERE__)
        << "verifying mesh@0x" << mesh
        << journal::endl;
    // NYI
    info 
        << journal::at(__HERE__)
        << "done verifying mesh@0x" << mesh
        << journal::endl;
    
    // return
    Py_INCREF(Py_None);
    return Py_None;
}


// $Id: verify.cc,v 1.1.1.1 2005/03/08 16:13:29 aivazis Exp $

// End of file
