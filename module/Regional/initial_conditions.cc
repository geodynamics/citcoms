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

    void convection_initial_temperature(struct All_variables*);
    void initial_pressure(struct All_variables*);
    void initial_velocity(struct All_variables*);
    void initial_viscosity(struct All_variables*);
    void report(struct All_variables*, char* str);

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



char pyCitcom_ic_initTemperature__doc__[] = "";
char pyCitcom_ic_initTemperature__name__[] = "initTemperature";

PyObject * pyCitcom_ic_initTemperature(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:initTemperature", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    report(E,"Initialize temperature field");
    convection_initial_temperature(E);

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
// $Id: initial_conditions.cc,v 1.1 2003/10/29 18:40:00 tan2 Exp $

// End of file
