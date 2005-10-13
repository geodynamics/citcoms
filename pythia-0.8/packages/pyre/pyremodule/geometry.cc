// -*- C++ -*-
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2005  All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 

#include <portinfo>
#include <Python.h>

#include "journal/debug.h"

#include "geometry.h"
#include "pyre/geometry/CanonicalMesh.h"

// types
typedef pyre::geometry::CanonicalMesh<double> mesh_t;

// helpers
extern "C" void _deleteMesh(void *);

// createMesh

char pyremodule_createMesh__doc__[] = "";
char pyremodule_createMesh__name__[] = "createMesh";

PyObject * pyremodule_createMesh(PyObject *, PyObject * args)
{
    int dim;
    int order;

    int ok = PyArg_ParseTuple(args, "ii:createMesh", &dim, &order);
    if (!ok) {
        return 0;
    }

    // create the mesh
    mesh_t * mesh = new mesh_t(dim, order);

    // report
    journal::debug_t debug("pyre.geometry");
    debug
        << journal::at(__HERE__)
        << "created mesh@" << mesh
        << journal::endl;

    // return
    return PyCObject_FromVoidPtr(mesh, _deleteMesh);
}

// mesh statistics
char pyremodule_statistics__doc__[] = "";
char pyremodule_statistics__name__[] = "statistics";

PyObject * pyremodule_statistics(PyObject *, PyObject * args)
{
    PyObject * py_mesh;

    int ok = PyArg_ParseTuple(args, "O:statistics", &py_mesh);
    if (!ok) {
        return 0;
    }

    mesh_t * mesh = (mesh_t *)PyCObject_AsVoidPtr(py_mesh);

    int dim, order, nVertices, nSimplices;

    dim = mesh->dim();
    order = mesh->order();
    nVertices = mesh->vertexCount();
    nSimplices = mesh->simplexCount();

    // return
    return Py_BuildValue("(iiii)", dim, order, nVertices, nSimplices);
}
    
// access to nodes
char pyremodule_vertex__doc__[] = "";
char pyremodule_vertex__name__[] = "vertex";

PyObject * pyremodule_vertex(PyObject *, PyObject * args)
{
    int vertexid;
    PyObject * py_mesh;

    int ok = PyArg_ParseTuple(args, "Oi:statistics", &py_mesh, &vertexid);
    if (!ok) {
        return 0;
    }

    mesh_t * mesh = (mesh_t *)PyCObject_AsVoidPtr(py_mesh);

    PyObject * result = PyTuple_New(mesh->dim());
    for (int axis = 0; axis < mesh->order(); ++axis) {
        PyTuple_SET_ITEM(result, axis, PyFloat_FromDouble(mesh->vertex(vertexid, axis)));
    }

    // return
    return result;
}

// access to simplices
char pyremodule_simplex__doc__[] = "";
char pyremodule_simplex__name__[] = "simplex";

PyObject * pyremodule_simplex(PyObject *, PyObject * args)
{
    int simplexid;
    PyObject * py_mesh;

    int ok = PyArg_ParseTuple(args, "Oi:statistics", &py_mesh, &simplexid);
    if (!ok) {
        return 0;
    }

    mesh_t * mesh = (mesh_t *)PyCObject_AsVoidPtr(py_mesh);

    PyObject * result = PyTuple_New(mesh->order());
    for (int vertex = 0; vertex < mesh->order(); ++vertex) {
        PyTuple_SET_ITEM(result, vertex, PyInt_FromLong(mesh->simplex(simplexid, vertex)));
    }

    // return
    return result;
}

// helpers
void _deleteMesh(void * object)
{
    mesh_t * mesh = (mesh_t *) object;

    journal::debug_t debug("pyre.geometry");
    debug
        << journal::at(__HERE__)
        << "deleting mesh@" << mesh
        << journal::endl;

    delete mesh;

    return;
}

// version
// $Id: geometry.cc,v 1.1.1.1 2005/03/08 16:13:52 aivazis Exp $

// End of file
