// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <Python.h>
#include <cstdio>

#include "advdiffu.h"

extern "C" {
#include "global_defs.h"
#include "citcom_init.h"
#include "advection_diffusion.h"
    void set_convection_defaults(struct All_variables *);

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


char pyRegional_set_convection_defaults__doc__[] = "";
char pyRegional_set_convection_defaults__name__[] = "set_convection_defaults";
PyObject * pyRegional_set_convection_defaults(PyObject *self, PyObject *args)
{

    E->control.CONVECTION = 1;
    set_convection_defaults(E);

    Py_INCREF(Py_None);
    return Py_None;
}


//////////////////////////////////////////////////////////////////////////



// version
// $Id: advdiffu.cc,v 1.4 2003/07/24 00:04:04 tan2 Exp $

// End of file
