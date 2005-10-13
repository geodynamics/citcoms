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
#include "operators.h"
#include "exceptions.h"
#include "support.h"

char pyacis_union__name__[] = "union";
char pyacis_union__doc__[] = "construct the union of two bodies";
PyObject * pyacis_union(PyObject *, PyObject *args)
{
    journal::debug_t info("acis.operators");

    PyObject * b1;
    PyObject * b2;

    int ok = PyArg_ParseTuple(args, "OO:union", &b1, &b2);
    if (!ok) {
        return 0;
    }

    BODY * tool = (BODY *)PyCObject_AsVoidPtr(b1);
    BODY * blank = (BODY *)PyCObject_AsVoidPtr(b2);

    outcome check = api_unite(tool, blank);
    if (!check.ok()) {
        throwACISError(check, "unite", pyacis_runtimeError);
        return 0;
    }

    info
        << journal::at(__HERE__)
        << "union@" << blank << ": tool=" << tool << ", blank=" << blank
        << journal::endl;

    return PyCObject_FromVoidPtr(blank, 0);
}


char pyacis_intersection__name__[] = "intersection";
char pyacis_intersection__doc__[] = "construct the intersection of two bodies";
PyObject * pyacis_intersection(PyObject *, PyObject *args)
{
    journal::debug_t info("acis.operators");

    PyObject * b1;
    PyObject * b2;

    int ok = PyArg_ParseTuple(args, "OO:intersection", &b1, &b2);
    if (!ok) {
        return 0;
    }

    BODY * tool = (BODY *)PyCObject_AsVoidPtr(b1);
    BODY * blank = (BODY *)PyCObject_AsVoidPtr(b2);

    outcome check = api_intersect(tool, blank);
    if (!check.ok()) {
        throwACISError(check, "intersect", pyacis_runtimeError);
        return 0;
    }

    info
        << journal::at(__HERE__)
        << "intersection@" << blank << ": tool=" << tool << ", blank=" << blank
        << journal::endl;

    return PyCObject_FromVoidPtr(blank, 0);
}


char pyacis_difference__name__[] = "difference";
char pyacis_difference__doc__[] = "construct the difference of two bodies";
PyObject * pyacis_difference(PyObject *, PyObject *args)
{
    journal::debug_t info("acis.operators");

    PyObject * b1;
    PyObject * b2;

    int ok = PyArg_ParseTuple(args, "OO:difference", &b1, &b2);
    if (!ok) {
        return 0;
    }

    // correct for the semantic discrepancey:
    // diference(a, b) == subtract(b, a)
    BODY * tool = (BODY *)PyCObject_AsVoidPtr(b2);
    BODY * blank = (BODY *)PyCObject_AsVoidPtr(b1);

    outcome check = api_subtract(tool, blank);
    if (!check.ok()) {
        throwACISError(check, "subtract", pyacis_runtimeError);
        return 0;
    }

    info
        << journal::at(__HERE__)
        << "difference@" << blank << ": tool=" << tool << ", blank=" << blank
        << journal::endl;

    return PyCObject_FromVoidPtr(blank, 0);
}

//
// $Id: operators.cc,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $

// End of file
