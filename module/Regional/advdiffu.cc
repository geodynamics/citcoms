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
  
  PG_timestep_Init(E);
  
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

char pyRegional_PG_timemarching_control__doc__[] = "";
char pyRegional_PG_timemarching_control__name__[] = "PG_timemarching_control";
PyObject * pyRegional_PG_timemarching_control(PyObject *self, PyObject *args)
{
  
  PG_timemarching_control(E);
  
  Py_INCREF(Py_None);
  return Py_None;
}

char pyRegional_PG_timestep_fini__doc__[] = "";
char pyRegional_PG_timestep_fini__name__[] = "PG_timestep_fini";
PyObject * pyRegional_PG_timestep_fini(PyObject *self, PyObject *args)
{
  
  PG_timestep_Fini(E);
  
  Py_INCREF(Py_None);
  return Py_None;
}

//////////////////////////////////////////////////////////////////////////



// version
// $Id: advdiffu.cc,v 1.1 2003/05/22 18:32:14 ces74 Exp $

// End of file
