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
#include "global_defs.h"
#include "citcom_init.h"

}

long getIntProperty(PyObject* properties, char* attribute);
double getFloatProperty(PyObject* properties, char* attribute);
void getStringProperty(PyObject* properties, char* attribute, char* value);
void getIntVecProperty(PyObject* properties, char* attribute, int* vector, int len);
void getFloatVecProperty(PyObject* properties, char* attribute, float* vector, int len);

//
//

char pyRegional_BC_set_prop__doc__[] = "";
char pyRegional_BC_set_prop__name__[] = "BC_set_prop";

PyObject * pyRegional_BC_set_prop(PyObject *self, PyObject *args)
{
    PyObject *properties;

    if (!PyArg_ParseTuple(args, "O:BC_set_prop", &properties))
        return NULL;

    std::cerr << "BC.inventories:" << std::endl;

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



char pyRegional_Visc_set_prop__doc__[] = "";
char pyRegional_Visc_set_prop__name__[] = "Visc_set_prop";

PyObject * pyRegional_Visc_set_prop(PyObject *self, PyObject *args)
{
    PyObject *properties;

    if (!PyArg_ParseTuple(args, "O:Visc_set_prop", &properties))
        return NULL;

    std::cerr << "Visc.inventories:" << std::endl;

    getStringProperty(properties, "Viscosity", E->viscosity.STRUCTURE);
    E->viscosity.RHEOL = getIntProperty(properties, "rheol");
    E->viscosity.smooth_cycles = getIntProperty(properties, "visc_smooth_method");
    E->viscosity.update_allowed = getIntProperty(properties, "VISC_UPDATE");
    int num_mat = getIntProperty(properties, "num_mat");

    if(num_mat > 40) {
	// max. allowed material types = 40
	std::cerr << "'num_mat' greater than allowed value, set to 40.";
	num_mat = 40;
    }
    E->viscosity.num_mat = num_mat;

    getFloatVecProperty(properties, "visc0",
			E->viscosity.N0, num_mat);

    E->viscosity.TDEPV = getIntProperty(properties, "TDEPV");
    getFloatVecProperty(properties, "viscE",
			E->viscosity.E, num_mat);
    getFloatVecProperty(properties, "viscT",
			E->viscosity.T, num_mat);

    E->viscosity.SDEPV = getIntProperty(properties, "SDEPV");
    E->viscosity.sdepv_misfit = getIntProperty(properties, "sdepv_misfit");
    getFloatVecProperty(properties, "sdepv_expt",
			E->viscosity.sdepv_expt, num_mat);

    E->viscosity.MIN = getIntProperty(properties, "VMIN");
    E->viscosity.min_value = getIntProperty(properties, "visc_min");

    E->viscosity.MAX = getIntProperty(properties, "VMAX");
    E->viscosity.max_value = getIntProperty(properties, "visc_max");

    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}




//==========================================================
// helper functions


long getIntProperty(PyObject* properties, char* attribute)
{
    std::cerr << '\t' << attribute << " = ";

    if(!PyObject_HasAttrString(properties, attribute)) {
	char errmsg[255];
	sprintf(errmsg, "no such attribute: %s", attribute);
	PyErr_SetString(PyExc_AttributeError, errmsg);
	return 0;
    }

    PyObject* prop = PyObject_GetAttrString(properties, attribute);
    if(!PyNumber_Check(prop)) {
	char errmsg[255];
	sprintf(errmsg, "'%s' is not a number", attribute);
	PyErr_SetString(PyExc_TypeError, errmsg);
	return 0;
    }

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
	return 0;
    }

    PyObject* prop = PyObject_GetAttrString(properties, attribute);
    if(!PyNumber_Check(prop)) {
	char errmsg[255];
	sprintf(errmsg, "'%s' is not a number", attribute);
	PyErr_SetString(PyExc_TypeError, errmsg);
	return 0;
    }

    double value = PyFloat_AsDouble(prop);
    std::cerr << value << std::endl;

    return value;
}



void getStringProperty(PyObject* properties, char* attribute, char* value)
{
    std::cerr << '\t' << attribute << " = ";

    if(!PyObject_HasAttrString(properties, attribute)) {
	char errmsg[255];
	sprintf(errmsg, "no such attribute: %s", attribute);
	PyErr_SetString(PyExc_AttributeError, errmsg);
	return;
    }

    PyObject* prop = PyObject_GetAttrString(properties, attribute);
    if(!PyString_Check(prop)) {
	char errmsg[255];
	sprintf(errmsg, "'%s' is not a string", attribute);
	PyErr_SetString(PyExc_TypeError, errmsg);
	return;
    }

    strcpy(value, PyString_AsString(prop));
    std::cerr << value << std::endl;

    return;
}



void getIntVecProperty(PyObject* properties, char* attribute, int* vector, int len)
{
    std::cerr << '\t' << attribute << " = ";

    if(!PyObject_HasAttrString(properties, attribute)) {
	char errmsg[255];
	sprintf(errmsg, "no such attribute: %s", attribute);
	PyErr_SetString(PyExc_AttributeError, errmsg);
	return;
    }

    // is it a sequence?
    PyObject* prop = PyObject_GetAttrString(properties, attribute);
    if(!PySequence_Check(prop)) {
	char errmsg[255];
	sprintf(errmsg, "'%s' is not a sequence", attribute);
	PyErr_SetString(PyExc_TypeError, errmsg);
	return;
    }

    // is it of length len?
    int n = PySequence_Size(prop);
    if(n < len) {
	char errmsg[255];
	sprintf(errmsg, "length of '%s' < %d", attribute, len);
	PyErr_SetString(PyExc_IndexError, errmsg);
	return;
    } else if(n > len) {
	char warnmsg[255];
	sprintf(warnmsg, "length of '%s' > %d", attribute, len);
	std::cerr << warnmsg << std::endl;
    }

    std::cerr << "[ ";
    for (int i=0; i<len; i++) {
	PyObject* item = PySequence_GetItem(prop, i);
	if(!item) {
	    char errmsg[255];
	    sprintf(errmsg, "can't get %s[%d]", attribute, i);
	    PyErr_SetString(PyExc_IndexError, errmsg);
	    return;
	}

	if(PyNumber_Check(item)) {
	    vector[i] = PyInt_AsLong(item);
	} else {
	    char errmsg[255];
	    sprintf(errmsg, "'%s[%d]' is not a number ", attribute, i);
	    PyErr_SetString(PyExc_TypeError, errmsg);
	    return;
	}
	std::cerr << vector[i] << ", ";
    }
    std::cerr << ']' << std::endl;

    return;
}



void getFloatVecProperty(PyObject* properties, char* attribute, float* vector, int len)
{
    std::cerr << '\t' << attribute << " = ";

    if(!PyObject_HasAttrString(properties, attribute)) {
	char errmsg[255];
	sprintf(errmsg, "no such attribute: %s", attribute);
	PyErr_SetString(PyExc_AttributeError, errmsg);
	return;
    }

    // is it a sequence?
    PyObject* prop = PyObject_GetAttrString(properties, attribute);
    if(!PySequence_Check(prop)) {
	char errmsg[255];
	sprintf(errmsg, "'%s' is not a sequence", attribute);
	PyErr_SetString(PyExc_TypeError, errmsg);
	return;
    }

    // is it of length len?
    int n = PySequence_Size(prop);
    if(n < len) {
	char errmsg[255];
	sprintf(errmsg, "length of '%s' < %d", attribute, len);
	PyErr_SetString(PyExc_IndexError, errmsg);
	return;
    } else if(n > len) {
	char warnmsg[255];
	sprintf(warnmsg, "length of '%s' > %d", attribute, len);
	std::cerr << warnmsg << std::endl;
    }

    std::cerr << "[ ";
    for (int i=0; i<len; i++) {
	PyObject* item = PySequence_GetItem(prop, i);
	if(!item) {
	    char errmsg[255];
	    sprintf(errmsg, "can't get %s[%d]", attribute, i);
	    PyErr_SetString(PyExc_IndexError, errmsg);
	    return;
	}

	if(PyNumber_Check(item)) {
	    vector[i] = PyFloat_AsDouble(item);
	} else {
	    char errmsg[255];
	    sprintf(errmsg, "'%s[%d]' is not a number ", attribute, i);
	    PyErr_SetString(PyExc_TypeError, errmsg);
	    return;
	}
	std::cerr << vector[i] << ", ";
    }
    std::cerr << ']' << std::endl;

    return;
}

