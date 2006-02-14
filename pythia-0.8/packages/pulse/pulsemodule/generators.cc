// -*- C++ -*-
//
//-----------------------------------------------------------------------------
//
//                              Michael A.G. Aivazis
//                       California Institute of Technology
//                       (C) 1998-2005  All Rights Reserved
//
// <LicenseText>
//
//-----------------------------------------------------------------------------
//

#include <portinfo>
#include <Python.h>

#include "generators.h"

#include "pyre/geometry/CanonicalMesh.h"
typedef pyre::geometry::CanonicalMesh<double> mesh_t;

#include "journal/debug.h"

// extern "C" void Py_MainDebugTrap();

// the routine that paints a heaviside pulse
char pypulse_heaviside__name__[] = "heaviside";
char pypulse_heaviside__doc__[] = "apply pressure values from a Heaviside pulse to the boundary";
PyObject * pypulse_heaviside(PyObject *, PyObject * args)
{
    // Py_MainDebugTrap();
    journal::debug_t info("pulse.generators");

    PyObject * py_mesh;
    PyObject * py_field;
    double amplitude;
    double r_x, r_y, r_z;
    double v_x, v_y, v_z;

    int ok = PyArg_ParseTuple(
        args, "OOd(ddd)(ddd):heaviside",
        &py_mesh, &py_field,
        &amplitude,
        &r_x, &r_y, &r_z,
        &v_x, &v_y, &v_z
        );

    if (!ok) {
        return 0;
    }

    mesh_t * mesh = static_cast<mesh_t *>(PyCObject_AsVoidPtr(py_mesh));
    double * pressure = static_cast<double *>(PyCObject_AsVoidPtr(py_field));

    info 
        << journal::at(__HERE__) 
        << "applying heaviside pressure pulse to field@" << pressure
        << " using mesh@0x" << mesh
        << journal::endl;

    for (size_t node = 0; node < mesh->vertexCount(); ++node) {

        double x = mesh->vertex(node, 0);
        double y = mesh->vertex(node, 1);
        double z = mesh->vertex(node, 2);

        double dot = (x-r_x)*v_x + (y-r_y)*v_y + (z-r_z)*v_z;

        if (dot >= 0.0)
            pressure[node] = 0.0;
        else
            pressure[node] = amplitude;
    }

    // return
    Py_INCREF(Py_None);
    return Py_None;
}


// pressure as a function of depth
char pypulse_bath__name__[] = "bath";
char pypulse_bath__doc__[] = "set the pressure as a function of depth";
PyObject * pypulse_bath(PyObject *, PyObject * args)
{
    // Py_MainDebugTrap();

    journal::debug_t info("pulse.generators");

    double fluidDensity;
    double surfacePressure;
    double surfacePosition;
    PyObject * py_mesh;
    PyObject * py_field;

    int ok = PyArg_ParseTuple(
        args, "OOddd:bath",
        &py_mesh,
        &py_field,
        &surfacePressure,
        &surfacePosition,
        &fluidDensity
        );

    if (!ok) {
        return 0;
    }

    mesh_t * mesh = static_cast<mesh_t *>(PyCObject_AsVoidPtr(py_mesh));
    double * pressure = static_cast<double *>(PyCObject_AsVoidPtr(py_field));

    info
        << journal::at(__HERE__) 
        << "applying bath pressure generator to mesh@0x" << mesh
        << journal::endl;

    for (size_t node = 0; node < mesh->vertexCount(); ++node) {
        double z = mesh->vertex(node, 2);

        if (z >= surfacePosition)
            pressure[node] = surfacePressure;
        else
            pressure[node] = 9.81*fluidDensity*(surfacePosition - z);
    }

    // return
    Py_INCREF(Py_None);
    return Py_None;
}


// $Id: generators.cc,v 1.1.1.1 2005/03/08 16:13:57 aivazis Exp $

// End of file
