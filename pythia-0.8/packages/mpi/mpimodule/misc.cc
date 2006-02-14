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
#include <mpi.h>

#include "misc.h"

// wtime

char pympi_wtime__doc__[] = "";
char pympi_wtime__name__[] = "time";
PyObject * pympi_wtime(PyObject *, PyObject *)
{
    return Py_BuildValue("d", MPI_Wtime());
}
    
// copyright

char pympi_copyright__doc__[] = "";
char pympi_copyright__name__[] = "copyright";

static char pympi_copyright_note[] = 
    "pythia.mpi module: Copyright (c) 1998-2005 Michael A.G. Aivazis";


PyObject * pympi_copyright(PyObject *, PyObject *)
{
    return Py_BuildValue("s", pympi_copyright_note);
}
    
// version
// $Id: misc.cc,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $

// End of file
