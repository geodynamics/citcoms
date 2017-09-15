/*
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//<LicenseText>
//
// CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
// Copyright (C) 2002-2005, California Institute of Technology.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//</LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

#include <Python.h>
#include <stdio.h>

#include "advdiffu.h"

#include "global_defs.h"
#include "advection_diffusion.h"


extern void set_convection_defaults(struct All_variables *);


char pyCitcom_PG_timestep_init__doc__[] = "";
char pyCitcom_PG_timestep_init__name__[] = "PG_timestep_init";
PyObject * pyCitcom_PG_timestep_init(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:PG_timestep_init", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    PG_timestep_init(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_PG_timestep_solve__doc__[] = "";
char pyCitcom_PG_timestep_solve__name__[] = "PG_timestep_solve";
PyObject * pyCitcom_PG_timestep_solve(PyObject *self, PyObject *args)
{
    PyObject *obj;
    double dt;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "Od:PG_timestep_solve", &obj, &dt))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    /* This function replaces some code in Citcom.c
     * If you modify here, make sure its counterpart
     * is modified as well */
    E->monitor.solution_cycles++;
    if(E->monitor.solution_cycles>E->control.print_convergence)
	E->control.print_convergence=1;

    /* Since dt may be modified in Pyre, we need to update
     * E->advection.timestep again */
    E->advection.timestep = dt;

    PG_timestep_solve(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_set_convection_defaults__doc__[] = "";
char pyCitcom_set_convection_defaults__name__[] = "set_convection_defaults";
PyObject * pyCitcom_set_convection_defaults(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:set_convection_defaults", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    set_convection_defaults(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_stable_timestep__doc__[] = "";
char pyCitcom_stable_timestep__name__[] = "stable_timestep";
PyObject * pyCitcom_stable_timestep(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:stable_timestep", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));


    std_timestep(E);

    return Py_BuildValue("d", E->advection.timestep);
}




/* $Id$ */

/* End of file */
