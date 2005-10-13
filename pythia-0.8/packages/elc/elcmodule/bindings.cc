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

#include "memory.h"        // allocators for various objects
#include "misc.h"          // miscellaneous methods
#include "verify.h"        // consistency checks

#if defined(WITH_MPI)
#include "via_mpi.h"       // transport via MPI
#endif

// the method table

struct PyMethodDef pyelc_methods[] = {
    // memory
    {pyelc_allocateField__name__, pyelc_allocateField,
     METH_VARARGS, pyelc_allocateField__doc__},

#if defined(WITH_MPI)
    // via_mpi
    {pyelc_sendBoundaryMPI__name__, pyelc_sendBoundaryMPI, 
     METH_VARARGS, pyelc_sendBoundaryMPI__doc__},
    {pyelc_receiveBoundaryMPI__name__, pyelc_receiveBoundaryMPI, 
     METH_VARARGS, pyelc_receiveBoundaryMPI__doc__},

    {pyelc_sendFieldMPI__name__, pyelc_sendFieldMPI, 
     METH_VARARGS, pyelc_sendFieldMPI__doc__},
    {pyelc_receiveFieldMPI__name__, pyelc_receiveFieldMPI, 
     METH_VARARGS, pyelc_receiveFieldMPI__doc__},
#endif

    // verify
    {pyelc_verify__name__, pyelc_verify,
     METH_VARARGS, pyelc_verify__doc__},

    // copyright
    {pyelc_copyright__name__, pyelc_copyright,
     METH_VARARGS, pyelc_copyright__doc__},


// Sentinel
    {0, 0}
};

// version
// $Id: bindings.cc,v 1.1.1.1 2005/03/08 16:13:29 aivazis Exp $

// End of file
