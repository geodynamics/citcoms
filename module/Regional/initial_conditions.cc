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
#include "exceptions.h"
#include "initial_conditions.h"

extern "C" {

#include "global_defs.h"

    void construct_tic_from_input(struct All_variables*);
    void initial_pressure(struct All_variables*);
    void initial_velocity(struct All_variables*);
    void initial_viscosity(struct All_variables*);
    void report(struct All_variables*, char* str);
    void restart_tic_from_file(struct All_variables*);

}


char pyCitcom_ic_constructTemperature__doc__[] = "";
char pyCitcom_ic_constructTemperature__name__[] = "constructTemperature";

PyObject * pyCitcom_ic_constructTemperature(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:constructTemperature", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    report(E,"Initialize temperature field");
    construct_tic_from_input(E);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_ic_restartTemperature__doc__[] = "";
char pyCitcom_ic_restartTemperature__name__[] = "restartTemperature";

PyObject * pyCitcom_ic_restartTemperature(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:restartTemperature", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    report(E,"Initialize temperature field");
    restart_tic_from_file(E);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_ic_initPressure__doc__[] = "";
char pyCitcom_ic_initPressure__name__[] = "initPressure";

PyObject * pyCitcom_ic_initPressure(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:initPressure", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

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

    if (!PyArg_ParseTuple(args, "O:initVelocity", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

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

    if (!PyArg_ParseTuple(args, "O:initViscosity", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    report(E,"Initialize viscosity field");
    initial_viscosity(E);

    Py_INCREF(Py_None);
    return Py_None;
}



// version
// $Id: initial_conditions.cc,v 1.2 2003/11/28 22:20:23 tan2 Exp $

// End of file
