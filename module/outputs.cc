// -*- C++ -*-
//
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
//

#include <portinfo>
#include <Python.h>
#include <cstdio>
#include <iostream>

#include "outputs.h"

#include "global_defs.h"
#include "output.h"


extern "C" {

    void output_finalize(struct  All_variables *E);

}


char pyCitcom_output__doc__[] = "";
char pyCitcom_output__name__[] = "output";

PyObject * pyCitcom_output(PyObject *self, PyObject *args)
{
    PyObject *obj;
    int cycles;

    if (!PyArg_ParseTuple(args, "Oi:output", &obj, &cycles))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    if(E->control.pseudo_free_surf)
	    if(E->mesh.topvbc==2)
		    output_pseudo_surf(E, cycles);
	    else
		    assert(0);
    else
	    (E->problem_output)(E, cycles);


    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_output_finalize__doc__[] = "";
char pyCitcom_output_finalize__name__[] = "output_finalize";

PyObject * pyCitcom_output_finalize(PyObject *self, PyObject *args)
{
    PyObject *obj;
    int cycles;

    if (!PyArg_ParseTuple(args, "O:output_finalize", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    output_finalize(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_output_time__doc__[] = "";
char pyCitcom_output_time__name__[] = "output_time";

PyObject * pyCitcom_output_time(PyObject *self, PyObject *args)
{
    PyObject *obj;
    int cycles;

    if (!PyArg_ParseTuple(args, "Oi:output_time", &obj, &cycles))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    output_time(E, cycles);


    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_output_pseudo_surf__doc__[] = "";
char pyCitcom_output_pseudo_surf__name__[] = "output_pseudo_surf";

PyObject * pyCitcom_output_pseudo_surf(PyObject *self, PyObject *args)
{
    PyObject *obj;
    int cycles;

    if (!PyArg_ParseTuple(args, "Oi:output_pseudo_surf", &obj, &cycles))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    output_pseudo_surf(E, cycles);

    Py_INCREF(Py_None);
    return Py_None;
}

// version
// $Id$

// End of file
