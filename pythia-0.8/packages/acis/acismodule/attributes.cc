// -*- C++ -*-
//
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2005 All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include "imports"

// Local

#include <ga_husk/api/ga_api.hxx>

#include "attributes.h"
#include "exceptions.h"
#include "support.h"


char pyacis_setAttributeInt__name__[] = "setAttributeInt";
char pyacis_setAttributeInt__doc__[] = "attach an integer attribute to a body";
PyObject * pyacis_setAttributeInt(PyObject *, PyObject * args)
{
    int value;
    char * name;
    PyObject * py_body;

    int ok = PyArg_ParseTuple(args, "Osi:setAttributeInt", &py_body, &name, &value);
    if (!ok) {
        return 0;
    }

    ENTITY * body = (ENTITY *) PyCObject_AsVoidPtr(py_body);

    outcome check = api_add_generic_named_attribute(body, name, value);
    if (!check.ok()) {
        throwACISError(check, "generic attribute", pyacis_runtimeError);
        return 0;
    }

    // return
    Py_INCREF(Py_None);
    return Py_None;
}


char pyacis_setAttributeDouble__name__[] = "setAttributeDouble";
char pyacis_setAttributeDouble__doc__[] = "attach a double attribute to a body";
PyObject * pyacis_setAttributeDouble(PyObject *, PyObject * args)
{
    char * name;
    double value;
    PyObject * py_body;

    int ok = PyArg_ParseTuple(args, "Osd:setAttributeDouble", &py_body, &name, &value);
    if (!ok) {
        return 0;
    }

    ENTITY * body = (ENTITY *) PyCObject_AsVoidPtr(py_body);

    outcome check = api_add_generic_named_attribute(body, name, value);
    if (!check.ok()) {
        throwACISError(check, "generic attribute", pyacis_runtimeError);
        return 0;
    }

    // return
    Py_INCREF(Py_None);
    return Py_None;
}


char pyacis_setAttributeString__name__[] = "setAttributeString";
char pyacis_setAttributeString__doc__[] = "attach a string attribute to a body";
PyObject * pyacis_setAttributeString(PyObject *, PyObject * args)
{
    char * name;
    char * value;
    PyObject * py_body;

    int ok = PyArg_ParseTuple(args, "Oss:setAttributeString", &py_body, &name, &value);
    if (!ok) {
        return 0;
    }

    ENTITY * body = (ENTITY *) PyCObject_AsVoidPtr(py_body);

    outcome check = api_add_generic_named_attribute(body, name, value);
    if (!check.ok()) {
        throwACISError(check, "generic attribute", pyacis_runtimeError);
        return 0;
    }

    // return
    Py_INCREF(Py_None);
    return Py_None;
}

// version
// $Id: attributes.cc,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $

// End of file
