/*
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

#include <Python.h>


#include "exceptions.h"
#include "stokes_solver.h"

#include "global_defs.h"
#include "drive_solvers.h"


void general_stokes_solver(struct All_variables *);
void general_stokes_solver_setup(struct All_variables*);
void set_cg_defaults(struct All_variables*);
void set_mg_defaults(struct All_variables*);
void solve_constrained_flow_iterative(struct All_variables*);


char pyCitcom_general_stokes_solver__doc__[] = "";
char pyCitcom_general_stokes_solver__name__[] = "general_stokes_solver";

PyObject * pyCitcom_general_stokes_solver(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:general_stokes_solver", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    if(E->control.pseudo_free_surf)
	    if(E->mesh.topvbc==2)
		    general_stokes_solver_pseudo_surf(E);
	    else
		    assert(0);
    else
	    general_stokes_solver(E);



    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_general_stokes_solver_setup__doc__[] = "";
char pyCitcom_general_stokes_solver_setup__name__[] = "general_stokes_solver_setup";

PyObject * pyCitcom_general_stokes_solver_setup(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:general_stokes_solver_setup", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    general_stokes_solver_setup(E);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_set_cg_defaults__doc__[] = "";
char pyCitcom_set_cg_defaults__name__[] = "set_cg_defaults";

PyObject * pyCitcom_set_cg_defaults(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:set_cg_defaults", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    set_cg_defaults(E);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_set_mg_defaults__doc__[] = "";
char pyCitcom_set_mg_defaults__name__[] = "set_mg_defaults";

PyObject * pyCitcom_set_mg_defaults(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:set_mg_defaults", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    set_mg_defaults(E);

    Py_INCREF(Py_None);
    return Py_None;
}


/* $Id: stokes_solver.c 16087 2009-12-08 21:43:15Z tan2 $ */

/* End of file */
