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
#include "initial_conditions.h"

#include "global_defs.h"


void initialize_material(struct All_variables*);
void initialize_tracers(struct All_variables*);
void init_composition(struct All_variables*);
void initial_pressure(struct All_variables*);
void initial_velocity(struct All_variables*);
void initial_viscosity(struct All_variables*);
void report(struct All_variables*, char* str);
void read_checkpoint(struct All_variables*);


char pyCitcom_ic_initialize_material__doc__[] = "";
char pyCitcom_ic_initialize_material__name__[] = "initialize_material";

PyObject * pyCitcom_ic_initialize_material(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:initialize_material", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    initialize_material(E);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_ic_init_tracer_composition__doc__[] = "";
char pyCitcom_ic_init_tracer_composition__name__[] = "init_tracer_composition";

PyObject * pyCitcom_ic_init_tracer_composition(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:init_tracer_composition", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    if (E->control.tracer==1) {
        initialize_tracers(E);

        if (E->composition.on)
            init_composition(E);
    }

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_ic_constructTemperature__doc__[] = "";
char pyCitcom_ic_constructTemperature__name__[] = "constructTemperature";

PyObject * pyCitcom_ic_constructTemperature(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:constructTemperature", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    (E->problem_initial_fields)(E);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_ic_initPressure__doc__[] = "";
char pyCitcom_ic_initPressure__name__[] = "initPressure";

PyObject * pyCitcom_ic_initPressure(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:initPressure", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    report(E,"Initialize pressure field");
    initial_pressure(E);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_ic_initVelocity__doc__[] = "";
char pyCitcom_ic_initVelocity__name__[] = "initVelocity";

PyObject * pyCitcom_ic_initVelocity(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:initVelocity", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    report(E,"Initialize velocity field");
    initial_velocity(E);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_ic_initViscosity__doc__[] = "";
char pyCitcom_ic_initViscosity__name__[] = "initViscosity";

PyObject * pyCitcom_ic_initViscosity(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:initViscosity", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    report(E,"Initialize viscosity field");
    initial_viscosity(E);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_ic_readCheckpoint__doc__[] = "";
char pyCitcom_ic_readCheckpoint__name__[] = "readCheckpoint";

PyObject * pyCitcom_ic_readCheckpoint(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:readCheckpoint", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    read_checkpoint(E);

    Py_INCREF(Py_None);
    return Py_None;
}



/* $Id$ */

/* End of file */
