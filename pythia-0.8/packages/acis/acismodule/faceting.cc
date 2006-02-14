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
#include "faceting.h"
#include "exceptions.h"
#include "support.h"

#include "Mesher.h"


// canonical mesh
#include "pyre/geometry/CanonicalMesh.h"
typedef pyre::geometry::CanonicalMesh<double> mesh_t;

// ACIS includes
#include <faceter/attribs/refine.hxx>
#include <faceter/attribs/af_enum.hxx>
#include <faceter/api/af_api.hxx>

char pyacis_facet__name__[] = "facet";
char pyacis_facet__doc__[] = "compute a triangulation of the given body";
PyObject * pyacis_facet(PyObject *, PyObject * args)
{
    journal::debug_t info("acis.faceting");

    PyObject * py_mesh;
    PyObject * py_body;
    PyObject * py_prop;

    int ok = PyArg_ParseTuple(args, "OOO:facet", &py_mesh, &py_body, &py_prop);
    if (!ok) {
        return 0;
    }

    BODY * body = (BODY *) PyCObject_AsVoidPtr(py_body);
    mesh_t * mesh = (mesh_t *) PyCObject_AsVoidPtr(py_mesh);

    info
        << journal::at(__HERE__)
        << "faceting body@" << body << ":"
        << journal::newline;

    PyObject * value;
    value = PyObject_GetAttrString(py_prop, "gridAspectRatio");
    if (!value) {
        return 0;
    }
    double aspectRatio = PyFloat_AsDouble(value);
    info << "    grid aspect ratio: " << aspectRatio
        << journal::newline;

    value = PyObject_GetAttrString(py_prop, "maximumEdgeLength");
    if (!value) {
        return 0;
    }

    double maxEdgeLength = PyFloat_AsDouble(value);
    info << "    maximum edge length: " << maxEdgeLength
        << journal::newline;

    value = PyObject_GetAttrString(py_prop, "maximumSurfaceTolerance");
    if (!value) {
        return 0;
    }

    double maxSurfaceTolerance = PyFloat_AsDouble(value);
    info 
        << "    maximum surface tolerance: " << maxSurfaceTolerance
        << journal::endl;

    // save the current mesh manager
    MESH_MANAGER * old;
    outcome check = api_get_mesh_manager(old);
    if (!check.ok()) {
        throwACISError(check, "facet", pyacis_runtimeError);
        return 0;
    }

    // install a new mesh manager
    Mesher * gipm = new Mesher;
    check = api_set_mesh_manager(gipm);
    if (!check.ok()) {
        throwACISError(check, "facet", pyacis_runtimeError);
        return 0;
    }

    // register the mesh with the mesh manager
    gipm->mesh(mesh);

    // build the refinement described by the faceting properties
    REFINEMENT * ref = new REFINEMENT;
    ref->set_grid_aspect_ratio(aspectRatio);
    ref->set_max_edge_length(maxEdgeLength);
    ref->set_surface_tol(maxSurfaceTolerance);

    ref->set_surf_mode(AF_SURF_ALL);
    ref->set_adjust_mode(AF_ADJUST_NONE);
    ref->set_triang_mode(AF_TRIANG_ALL);
    api_set_default_refinement(ref);

    // apply the body transformations to its geometry
    // setting the body transformation to NULL forces the geometry to change
    check = api_change_body_trans(body, NULL);
    if (!check.ok()) {
        throwACISError(check, "facet", pyacis_runtimeError);
        return 0;
    }

    // facet it
    info 
        << journal::at(__HERE__)
        << "faceting body@" << body
        << journal::endl;

    check = api_facet_entity(body);
    if (!check.ok()) {
        throwACISError(check, "facet", pyacis_runtimeError);
        return 0;
    }
    info 
        << journal::at(__HERE__) << "done faceting body@" << body << journal::endl;

#if 0
    // print out the mesh
    size_t sz = 4*sizeof(int) 
        + 3*(gipm->nodes()*sizeof(double) + gipm->triangles()*sizeof(int))
        + gipm->nodes()*sizeof(double);
    char * buffer = new char[sz];
    info 
        << journal::at(__HERE__)
        << "packing triangulation: " << sz << " bytes in buffer@" << (void *)buffer
        << journal::endl;
    gipm->pack(buffer, sz);
    info 
        << journal::at(__HERE__)
        << "done packing triangulation"
        << journal::endl;

    if (!check.ok()) {
        throwACISError(check, "facet", pyacis_runtimeError);
        return 0;
    }

    // restore original mesh manager
    check = api_set_mesh_manager(old);
    if (!check.ok()) {
        throwACISError(check, "facet", pyacis_runtimeError);
        return 0;
    }

    delete gipm;

    // return
    PyObject * packet = PyString_FromStringAndSize(buffer, sz);
    delete [] buffer;

    return packet;
#else
    // return
    Py_INCREF(Py_None);
    return Py_None;

#endif
}

// version
// $Id: faceting.cc,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $

// End of file
