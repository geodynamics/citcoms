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

// constructors for the primitive ACIS solids

#include "imports"

// local
#include "support.h"
#include "exceptions.h"
#include "solids.h"


char pyacis_block__name__[] = "block";
char pyacis_block__doc__[] = "construct a block";
PyObject * pyacis_block(PyObject *, PyObject * args)
{
    journal::debug_t info("acis.solids");

    double x, y, z;
    int ok = PyArg_ParseTuple(args, "(ddd)", &x, &y, &z);
    if (!ok) {
        return 0;
    }
        
    BODY * acisRep = 0;
    position across(x, y, z);
    position origin(0.0, 0.0, 0.0);

    outcome check = api_solid_block(origin, across, acisRep);
    if (!check.ok()) {
        throwACISError(check, "block", pyacis_runtimeError);
        return 0;
    }

    info
        << journal::at(__HERE__)
        << "block@" << acisRep 
        << ": diagonal=(" << x << ", " << y << ", " << z << ")"
        << journal::endl;

    return PyCObject_FromVoidPtr(acisRep, 0);
}


char pyacis_cone__name__[] = "cone";
char pyacis_cone__doc__[] = "construct a cone";
PyObject * pyacis_cone(PyObject *, PyObject * args)
{
    journal::debug_t info("acis.solids");

    double height;
    double topRadius;
    double bottomRadius;

    int ok = PyArg_ParseTuple(args, "ddd", &bottomRadius, &topRadius, &height);
    if (!ok) {
        return 0;
    }

    position top(0,0,height);
    position bottom(0,0,0);

    BODY * acisRep = 0;
    outcome check = api_solid_cylinder_cone(
        bottom, top, bottomRadius, bottomRadius, topRadius, 0, acisRep);
    if (!check.ok()) {
        throwACISError(check, "cone", pyacis_runtimeError);
        return 0;
    }

    info
        << journal::at(__HERE__)
        << "cone@" << acisRep 
        << ": top=" << topRadius << ", bottom=" << bottomRadius << ", height=" << height
        << journal::endl;

    return PyCObject_FromVoidPtr(acisRep, 0);
}


char pyacis_cylinder__name__[] = "cylinder";
char pyacis_cylinder__doc__[] = "construct a cylinder";
PyObject * pyacis_cylinder(PyObject *, PyObject * args)
{
    journal::debug_t info("acis.solids");

    double height;
    double radius;

    int ok = PyArg_ParseTuple(args, "dd", &radius, &height);
    if (!ok) {
        return 0;
    }

    position top(0,0,height);
    position bottom(0,0,0);

    BODY * acisRep = 0;
    outcome check = api_solid_cylinder_cone(bottom, top, radius, radius, radius, 0, acisRep);
    if (!check.ok()) {
        throwACISError(check, "cylinder", pyacis_runtimeError);
        return 0;
    }

    info
        << journal::at(__HERE__)
        << "cylinder@" << acisRep 
        << ": radius=" << radius << ", height=" << height
        << journal::endl;

    return PyCObject_FromVoidPtr(acisRep, 0);
}


char pyacis_prism__name__[] = "prism";
char pyacis_prism__doc__[] = "construct a prism";
PyObject * pyacis_prism(PyObject *, PyObject * args)
{
    Py_INCREF(Py_None);
    return Py_None;
}


char pyacis_pyramid__name__[] = "pyramid";
char pyacis_pyramid__doc__[] = "construct a pyramid";
PyObject * pyacis_pyramid(PyObject *, PyObject * args)
{
    Py_INCREF(Py_None);
    return Py_None;
}


char pyacis_sphere__name__[] = "sphere";
char pyacis_sphere__doc__[] = "construct a sphere";
PyObject * pyacis_sphere(PyObject *, PyObject * args)
{
    journal::debug_t info("acis.solids");

    double radius;
    int ok = PyArg_ParseTuple(args, "d", &radius);
    if (!ok) {
        return 0;
    }
  
    BODY * acisRep = 0;
    outcome check = api_make_sphere(radius, acisRep);
    if (!check.ok()) {
        throwACISError(check, "sphere", pyacis_runtimeError);
        return 0;
    }

    info
        << journal::at(__HERE__)
        << "sphere@" << acisRep 
        << ": radius=" << radius
        << journal::endl;

    return PyCObject_FromVoidPtr(acisRep, 0);
}


char pyacis_torus__name__[] = "torus";
char pyacis_torus__doc__[] = "construct a torus";
PyObject * pyacis_torus(PyObject *, PyObject * args)
{
    journal::debug_t info("acis.solids");

    double majorRadius;
    double minorRadius;
    int ok = PyArg_ParseTuple(args, "dd", &majorRadius, &minorRadius);
    if (!ok) {
        return 0;
    }
  
    BODY * acisRep = 0;
    outcome check = api_make_torus(majorRadius, minorRadius, acisRep);
    if (!check.ok()) {
        throwACISError(check, "torus", pyacis_runtimeError);
        return 0;
    }

    info
        << journal::at(__HERE__)
        << "torus@" << acisRep 
        << ": major-radius=" << majorRadius << ", minor-radius=" << minorRadius
        << journal::endl;

    return PyCObject_FromVoidPtr(acisRep, 0);
}

char pyacis_generalizedCone__name__[] = "generalizedCone";
char pyacis_generalizedCone__doc__[] = "construct a generalizedCone";
PyObject * pyacis_generalizedCone(PyObject *, PyObject * args)
{
    journal::debug_t info("acis.solids");

    double major;
    double minor;
    double scale;
    double height;

    int ok = PyArg_ParseTuple(args, "dddd", &major, &minor, &scale, &height);
    if (!ok) {
        return 0;
    }

    position top(0,0,height);
    position bottom(0,0,0);

    BODY * acisRep = 0;
    outcome check = api_solid_cylinder_cone(bottom, top, major, minor, major*scale, 0, acisRep);

    if (!check.ok()) {
        throwACISError(check, "generalizedCone", pyacis_runtimeError);
        return 0;
    }

    info
        << journal::at(__HERE__)
        << "generalizedCone@" << acisRep 
        << ": major=" << major << ", minor=" << minor
        << ", scale=" << scale << ", height=" << height
        << journal::endl;

    return PyCObject_FromVoidPtr(acisRep, 0);
}



// version
// $Id: solids.cc,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $

// End of file
