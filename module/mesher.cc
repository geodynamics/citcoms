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
#include "mesher.h"

#include "global_defs.h"
#include "instructions.h"
#include "parallel_util.h"


char pyCitcom_full_sphere_launch__doc__[] = "";
char pyCitcom_full_sphere_launch__name__[] = "full_sphere_launch";

PyObject * pyCitcom_full_sphere_launch(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:full_sphere_launch", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    initial_mesh_solver_setup(E);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_regional_sphere_launch__doc__[] = "";
char pyCitcom_regional_sphere_launch__name__[] = "regional_sphere_launch";

PyObject * pyCitcom_regional_sphere_launch(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:regional_sphere_launch", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    initial_mesh_solver_setup(E);

    Py_INCREF(Py_None);
    return Py_None;
}



/* $Id$ */

/* End of file */
