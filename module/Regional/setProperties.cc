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


void getStringProperty(PyObject* properties, char* attribute, char* value);

template <class T>
void getScalarProperty(PyObject* properties, char* attribute, T& value);

template <class T>
void getVectorProperty(PyObject* properties, char* attribute,
		       T* vector, int len);

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

    getScalarProperty(properties, "topvbc", E->mesh.topvbc);
    getScalarProperty(properties, "topvbxval", E->control.VBXtopval);
    getScalarProperty(properties, "topvbyval", E->control.VBYtopval);

    getScalarProperty(properties, "botvbc", E->mesh.botvbc);
    getScalarProperty(properties, "botvbxval", E->control.VBXbotval);
    getScalarProperty(properties, "botvbyval", E->control.VBYbotval);

    getScalarProperty(properties, "toptbc", E->mesh.toptbc);
    getScalarProperty(properties, "toptbcval", E->control.TBCtopval);

    getScalarProperty(properties, "bottbc", E->mesh.bottbc);
    getScalarProperty(properties, "bottbcval", E->control.TBCbotval);

    getScalarProperty(properties, "temperature_bound_adj", E->control.temperature_bound_adj);
    getScalarProperty(properties, "depth_bound_adj", E->control.depth_bound_adj);
    getScalarProperty(properties, "width_bound_adj", E->control.width_bound_adj);


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

    getScalarProperty(properties, "rheol", E->viscosity.RHEOL);


    getScalarProperty(properties, "visc_smooth_method", E->viscosity.smooth_cycles);
    getScalarProperty(properties, "VISC_UPDATE", E->viscosity.update_allowed);

    int num_mat;
    getScalarProperty(properties, "num_mat", num_mat);
    if(num_mat > 40) {
	// max. allowed material types = 40
	std::cerr << "'num_mat' greater than allowed value, set to 40.";
	num_mat = 40;
    }
    E->viscosity.num_mat = num_mat;

    getVectorProperty(properties, "visc0",
			E->viscosity.N0, num_mat);

    getScalarProperty(properties, "TDEPV", E->viscosity.TDEPV);
    getVectorProperty(properties, "viscE",
			E->viscosity.E, num_mat);
    getVectorProperty(properties, "viscT",
			E->viscosity.T, num_mat);

    getScalarProperty(properties, "SDEPV", E->viscosity.SDEPV);
    getScalarProperty(properties, "sdepv_misfit", E->viscosity.sdepv_misfit);
    getVectorProperty(properties, "sdepv_expt",
			E->viscosity.sdepv_expt, num_mat);

    getScalarProperty(properties, "VMIN", E->viscosity.MIN);
    getScalarProperty(properties, "visc_min", E->viscosity.min_value);

    getScalarProperty(properties, "VMAX", E->viscosity.MAX);
    getScalarProperty(properties, "visc_max", E->viscosity.max_value);

    if (PyErr_Occurred())
	return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}




//==========================================================
// helper functions


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



template <class T>
void getScalarProperty(PyObject* properties, char* attribute, T& value)
{
    std::cerr << '\t' << attribute << " = ";

    if(!PyObject_HasAttrString(properties, attribute)) {
	char errmsg[255];
	sprintf(errmsg, "no such attribute: %s", attribute);
	PyErr_SetString(PyExc_AttributeError, errmsg);
	return;
    }

    PyObject* prop = PyObject_GetAttrString(properties, attribute);
    if(!PyNumber_Check(prop)) {
	char errmsg[255];
	sprintf(errmsg, "'%s' is not a number", attribute);
	PyErr_SetString(PyExc_TypeError, errmsg);
	return;
    }

    value = static_cast<T>(PyFloat_AsDouble(prop));
    std::cerr << value << std::endl;

    return;
}



template <class T>
void getVectorProperty(PyObject* properties, char* attribute,
		       T* vector, const int len)
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
	    vector[i] = static_cast<T>(PyFloat_AsDouble(item));
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

