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
#include <iostream>
#include <string>

#include "setProperties.h"

extern "C" {
#include "mpi.h"
#include "global_defs.h"
#include "citcom_init.h"

}

//
// helper functions

long getIntProperty(PyObject* properties, char* attribute)
{
    std::cerr << '\t' << attribute << " = ";
    
    if(!PyObject_HasAttrString(properties, attribute)) {
	char errmsg[255];
	sprintf(errmsg, "no such attribute: %s", attribute);
	PyErr_SetString(PyExc_AttributeError, errmsg);
	//std::cerr << errmsg << "\n";
	return 0;
    }
    
    PyObject* prop = PyObject_GetAttrString(properties, attribute);
    long value = PyInt_AsLong(prop);
    std::cerr << value << std::endl;

    return value;
}


double getFloatProperty(PyObject* properties, char* attribute)
{
    std::cerr << '\t' << attribute << " = ";
    
    if(!PyObject_HasAttrString(properties, attribute)) {
	char errmsg[255];
	sprintf(errmsg, "no such attribute: %s", attribute);
	PyErr_SetString(PyExc_AttributeError, errmsg);
	//std::cerr << errmsg << "\n";
	return 0;
    }
    
    PyObject* prop = PyObject_GetAttrString(properties, attribute);
    double value = PyFloat_AsDouble(prop);
    std::cerr << value << std::endl;

    return value;
}



//
//

char pyRegional_BC_set_prop__doc__[] = "";
char pyRegional_BC_set_prop__name__[] = "BC_set_prop";

PyObject * pyRegional_BC_set_prop(PyObject *self, PyObject *args)
{
    PyObject *properties;

    if (!PyArg_ParseTuple(args, "O:BC_set_prop", &properties))
        return NULL;

    std::cerr << "BC.properites:" << std::endl;

    E->mesh.topvbc = getIntProperty(properties, "topvbc");
    E->control.VBXtopval = getFloatProperty(properties, "topvbxval");
    E->control.VBYtopval = getFloatProperty(properties, "topvbyval");

    E->mesh.botvbc = getIntProperty(properties, "botvbc");
    E->control.VBXbotval = getFloatProperty(properties, "botvbxval");
    E->control.VBYbotval = getFloatProperty(properties, "botvbyval");

    E->mesh.toptbc = getIntProperty(properties, "toptbc");
    E->control.TBCtopval = getFloatProperty(properties, "toptbcval");

    E->mesh.bottbc = getIntProperty(properties, "bottbc");
    E->control.TBCbotval = getFloatProperty(properties, "bottbcval");

    E->control.temperature_bound_adj = getIntProperty(properties, "temperature_bound_adj");
    E->control.depth_bound_adj = getFloatProperty(properties, "depth_bound_adj");
    E->control.width_bound_adj = getFloatProperty(properties, "width_bound_adj");


    if (PyErr_Occurred())
	return NULL;
    
    Py_INCREF(Py_None);
    return Py_None;
}

