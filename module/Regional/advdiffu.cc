// -*- C++ -*-
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 

#include <portinfo>
#include <Python.h>
#include <cstdio>

#include "advdiffu.h"

extern "C" {
#include "mpi.h"
#include "global_defs.h"
#include "citcom_init.h"
#include "advection_diffusion.h"
}

char pyRegional_PG_timestep_init__doc__[] = "";
char pyRegional_PG_timestep_init__name__[] = "PG_timestep_init";
PyObject * pyRegional_PG_timestep_init(PyObject *self, PyObject *args)
{
  
  PG_timestep_init(E);
  
  Py_INCREF(Py_None);
  return Py_None;
}

char pyRegional_PG_timestep_solve__doc__[] = "";
char pyRegional_PG_timestep_solve__name__[] = "PG_timestep_solve";
PyObject * pyRegional_PG_timestep_solve(PyObject *self, PyObject *args)
{
  
  PG_timestep_solve(E);
  
  Py_INCREF(Py_None);
  return Py_None;
}


//////////////////////////////////////////////////////////////////////////



// version
// $Id: advdiffu.cc,v 1.3 2003/05/23 05:12:29 ces74 Exp $

// End of file
