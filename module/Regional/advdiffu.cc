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

char pyCitcom_PG_timestep_init__doc__[] = "";
char pyCitcom_PG_timestep_init__name__[] = "PG_timestep_init";
PyObject * pyCitcom_PG_timestep_init(PyObject *self, PyObject *args)
{

    PG_timestep_init(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_PG_timestep_solve__doc__[] = "";
char pyCitcom_PG_timestep_solve__name__[] = "PG_timestep_solve";
PyObject * pyCitcom_PG_timestep_solve(PyObject *self, PyObject *args)
{

    E->monitor.solution_cycles++;
    if(E->monitor.solution_cycles>E->control.print_convergence)
	E->control.print_convergence=1;

    PG_timestep_solve(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_set_convection_defaults__doc__[] = "";
char pyCitcom_set_convection_defaults__name__[] = "set_convection_defaults";
PyObject * pyCitcom_set_convection_defaults(PyObject *self, PyObject *args)
{

    E->control.CONVECTION = 1;
    set_convection_defaults(E);

    // copied from advection_diffusion_parameters()
    E->advection.total_timesteps = 1; 
    E->advection.sub_iterations = 1;
    E->advection.last_sub_iterations = 1;
    E->advection.gamma = 0.5;
    E->monitor.T_maxvaried = 1.05;

    Py_INCREF(Py_None);
    return Py_None;
}


//////////////////////////////////////////////////////////////////////////



// version
// $Id: advdiffu.cc,v 1.6 2003/08/01 22:53:50 tan2 Exp $

// End of file
