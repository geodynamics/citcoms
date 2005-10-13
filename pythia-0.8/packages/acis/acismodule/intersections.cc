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

#include "intersections.h"
#include "exceptions.h"
#include "support.h"


char pyacis_facesIntersectQ__name__[] = "facesIntersectQ";
char pyacis_facesIntersectQ__doc__[] = "check whether two faces intersect";
PyObject * pyacis_facesIntersectQ(PyObject *, PyObject * args)
{
    PyObject * py_f1;
    PyObject * py_f2;

    int ok = PyArg_ParseTuple(args, "OO:facesIntersectQ", &py_f1, &py_f2);
    if (!ok) {
        return 0;
    }

    int flag = 0;
    FACE * f1 = (FACE *) PyCObject_AsVoidPtr(py_f1);
    FACE * f2 = (FACE *) PyCObject_AsVoidPtr(py_f2);


    API_NOP_BEGIN;

    BODY * body = 0;
    outcome check = api_fafa_int(f1, f2, body);
    if (check.ok() && body) {
        flag = 1;
    }

    API_NOP_END;

    // return
    return Py_BuildValue("i", flag);
}


char pyacis_bodiesIntersectQ__name__[] = "bodiesIntersectQ";
char pyacis_bodiesIntersectQ__doc__[] = "check whether two bodies intersect";
PyObject * pyacis_bodiesIntersectQ(PyObject *, PyObject * args)
{
    PyObject * py_b1;
    PyObject * py_b2;

    int ok = PyArg_ParseTuple(args, "OO:bodiesIntersectQ", &py_b1, &py_b2);
    if (!ok) {
        return 0;
    }

    int flag = 0;
    BODY * b1 = (BODY *) PyCObject_AsVoidPtr(py_b1);
    BODY * b2 = (BODY *) PyCObject_AsVoidPtr(py_b2);


    API_NOP_BEGIN;

    BODY * body = 0;
    outcome check = api_slice(b1, b2, *(unit_vector*)0, body);
    if (check.ok() && body) {
        flag = 1;
    }

    API_NOP_END;

    // return
    return Py_BuildValue("i", flag);
}


// version
// $Id: intersections.cc,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $

// End of file
