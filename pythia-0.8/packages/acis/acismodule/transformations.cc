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

// local
#include "transformations.h"
#include "exceptions.h"
#include "support.h"


char pyacis_dilation__name__[] = "dilation";
char pyacis_dilation__doc__[] = "scale a body by a factor";
PyObject * pyacis_dilation(PyObject *, PyObject *args)
{
    journal::debug_t info("acis.transformations");

    double scale;
    PyObject * py_body;

    int ok = PyArg_ParseTuple(args, "Od:dilation", &py_body, &scale);
    if (!ok) {
        return 0;
    }

    BODY * body = (BODY *) PyCObject_AsVoidPtr(py_body);

    transf dilation = scale_transf(scale);

    outcome check = api_transform_entity(body, dilation);
    if (!check.ok()) {
        throwACISError(check, "dilate", pyacis_runtimeError);
        return 0;
    }

    info
        << journal::at(__HERE__)
        << "dilation@" << body 
        << ": scale=" << scale
        << journal::endl;

    return PyCObject_FromVoidPtr(body, 0);
}


char pyacis_reflection__name__[] = "reflection";
char pyacis_reflection__doc__[] = "reflect a body about a vector";
PyObject * pyacis_reflection(PyObject *, PyObject *args)
{
    journal::debug_t info("acis.transformations");

    double x, y, z;
    PyObject * py_body;

    int ok = PyArg_ParseTuple(args, "O(ddd):reflection", &py_body, &x, &y, &z);
    if (!ok) {
        return 0;
    }

    BODY * body = (BODY *) PyCObject_AsVoidPtr(py_body);

    transf reflection = reflect_transf(vector(x, y, z));

    outcome check = api_transform_entity(body, reflection);
    if (!check.ok()) {
        throwACISError(check, "reflection", pyacis_runtimeError);
        return 0;
    }

    info
        << journal::at(__HERE__)
        << "reflection@" << body 
        << ": vector=(" << x << ", " << y << ", " << z << ")"
        << journal::endl;

    return PyCObject_FromVoidPtr(body, 0);
}


char pyacis_rotation__name__[] = "rotation";
char pyacis_rotation__doc__[] = "rotate a body by an angle about a vector";
PyObject * pyacis_rotation(PyObject *, PyObject *args)
{
    journal::debug_t info("acis.transformations");

    double angle;
    double x, y, z;
    PyObject * py_body;

    int ok = PyArg_ParseTuple(args, "Od(ddd):rotation", &py_body, &angle, &x, &y, &z);
    if (!ok) {
        return 0;
    }

    BODY * body = (BODY *) PyCObject_AsVoidPtr(py_body);

    transf rotation = rotate_transf(angle, vector(x, y, z));

    outcome check = api_transform_entity(body, rotation);
    if (!check.ok()) {
        throwACISError(check, "rotation", pyacis_runtimeError);
        return 0;
    }

    info
        << journal::at(__HERE__)
        << "rotation@" << body 
        << ": vector=(" << x << ", " << y << ", " << z << ")"
        << ", angle=" << angle
        << journal::endl;

    return PyCObject_FromVoidPtr(body, 0);
}


char pyacis_translation__name__[] = "translation";
char pyacis_translation__doc__[] = "translate a body by a vector";
PyObject * pyacis_translation(PyObject *, PyObject *args)
{
#if 1
    journal::debug_t info("acis.transformations");
#endif

    double x, y, z;
    PyObject * py_body;

    int ok = PyArg_ParseTuple(args, "O(ddd):translation", &py_body, &x, &y, &z);
    if (!ok) {
        return 0;
    }

    BODY * body = (BODY *) PyCObject_AsVoidPtr(py_body);

    transf translation = translate_transf(vector(x, y, z));

    outcome check = api_transform_entity(body, translation);
    if (!check.ok()) {
        throwACISError(check, "translation", pyacis_runtimeError);
        return 0;
    }

#if 1
    info
        << journal::at(__HERE__)
        << "translation@" << body 
        << ": vector=(" << x << ", " << y << ", " << z << ")"
        << journal::endl;
#endif

    return PyCObject_FromVoidPtr(body, 0);
}


char pyacis_reversal__name__[] = "reversal";
char pyacis_reversal__doc__[] = "reverse a body";
PyObject * pyacis_reversal(PyObject *, PyObject *args)
{
    journal::debug_t info("acis.transformations");

    PyObject * py_body;

    int ok = PyArg_ParseTuple(args, "O:reversal", &py_body);
    if (!ok) {
        return 0;
    }

    BODY * body = (BODY *) PyCObject_AsVoidPtr(py_body);

    outcome check = api_reverse_body(body);
    if (!check.ok()) {
        throwACISError(check, "reversal", pyacis_runtimeError);
        return 0;
    }

    info
        << journal::at(__HERE__)
        << "reversal@" << body 
        << journal::endl;

    return PyCObject_FromVoidPtr(body, 0);
}

// version
// $Id: transformations.cc,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $

// End of file
