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

#include <kernel/kerndata/lists/lists.hxx>
#include <baseutil/vector/position.hxx>

// 

#include "entities.h"
#include "exceptions.h"
#include "support.h"


char pyacis_box__name__[] = "box";
char pyacis_box__doc__[] = "compute and return the bounding box of an entity";
PyObject * pyacis_box(PyObject *, PyObject * args)
{
    PyObject * py_entity;

    int ok = PyArg_ParseTuple(args, "O:box", &py_entity);
    if (!ok) {
        return 0;
    }

    ENTITY * entity = (ENTITY *) PyCObject_AsVoidPtr(py_entity);

    ENTITY_LIST elist;
    position low;
    position high;

    elist.add(entity);
    outcome check = api_get_entity_box(elist, 0, low, high);
    if (!check.ok()) {
        throwACISError(check, "box", pyacis_runtimeError);
        return 0;
    }

    double x_min = low.x();
    double y_min = low.y();
    double z_min = low.z();
    double x_max = high.x();
    double y_max = high.y();
    double z_max = high.z();

    // return
    return Py_BuildValue("((ddd)(ddd))", x_min, y_min, z_min, x_max, y_max, z_max);
}


char pyacis_faces__name__[] = "faces";
char pyacis_faces__doc__[] = "return a list of the faces of a body";
PyObject * pyacis_faces(PyObject *, PyObject * args)
{
    PyObject * py_entity;

    int ok = PyArg_ParseTuple(args, "O:faces", &py_entity);
    if (!ok) {
        return 0;
    }

    ENTITY * entity = (ENTITY *) PyCObject_AsVoidPtr(py_entity);

    ENTITY_LIST facelist;
    outcome check = api_get_faces(entity, facelist);
    if (!check.ok()) {
        throwACISError(check, "faces", pyacis_runtimeError);
        return 0;
    }

    int faces = facelist.count();
    PyObject * facetuple = PyTuple_New(faces);

    facelist.init();
    for (int index = 0; index < faces; ++index) {
        PyTuple_SET_ITEM(facetuple, index, PyCObject_FromVoidPtr(facelist.next(), 0));
    }

    // return
    return facetuple;
}


char pyacis_distance__name__[] = "distance";
char pyacis_distance__doc__[] = "compute the distance between the two entities";
PyObject * pyacis_distance(PyObject *, PyObject * args)
{
    PyObject * py_e1;
    PyObject * py_e2;

    int ok = PyArg_ParseTuple(args, "OO:distance", &py_e1, &py_e2);
    if (!ok) {
        return 0;
    }

    double distance;
    position p1;
    position p2;
    ENTITY * e1 = (ENTITY *) PyCObject_AsVoidPtr(py_e1);
    ENTITY * e2 = (ENTITY *) PyCObject_AsVoidPtr(py_e2);

    outcome check = api_entity_entity_distance(e1, e2, p1, p2, distance);
    if (!check.ok()) {
        throwACISError(check, "faces", pyacis_runtimeError);
        return 0;
    }

    // return
    return Py_BuildValue(
        "d(ddd)(ddd)",
        distance, p1.x(), p1.y(), p1.z(), p2.x(), p2.y(), p2.z());
}


char pyacis_touch__name__[] = "touch";
char pyacis_touch__doc__[] = "return true if the two entities touch";
PyObject * pyacis_touch(PyObject *, PyObject * args)
{
    PyObject * py_e1;
    PyObject * py_e2;

    int ok = PyArg_ParseTuple(args, "OO:touch", &py_e1, &py_e2);
    if (!ok) {
        return 0;
    }

    ENTITY * e1 = (ENTITY *) PyCObject_AsVoidPtr(py_e1);
    ENTITY * e2 = (ENTITY *) PyCObject_AsVoidPtr(py_e2);

    logical flag;
    outcome check = api_entity_entity_touch(e1, e2, flag);
    if (!check.ok()) {
        throwACISError(check, "faces", pyacis_runtimeError);
        return 0;
    }

    // return
    return Py_BuildValue("i", flag);
}

// version
// $Id: entities.cc,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $

// End of file
