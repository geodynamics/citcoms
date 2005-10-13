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
#include "exceptions.h"
#include "faceting.h"
#include "support.h"

// ACIS includes
#include <faceter/attribs/refine.hxx>
#include <faceter/attribs/af_enum.hxx>
#include <faceter/api/af_api.hxx>
#include <meshhusk/api/meshapi.hxx>

#include <kernel/kerndata/geometry/getbox.hxx> 

char pyacis_mesh__name__[] = "mesh";
char pyacis_mesh__doc__[] = "compute a triangulation of the given body";
PyObject * pyacis_mesh(PyObject *, PyObject * args)
{
    journal::debug_t info("acis.meshing");

    PyObject * py_body;
    PyObject * py_prop;

    int ok = PyArg_ParseTuple(args, "OO:mesh", &py_body, &py_prop);
    if (!ok) {
        return 0;
    }

    BODY * body = (BODY *) PyCObject_AsVoidPtr(py_body);

    if (PyErr_Occurred()) {
        return 0;
    }

    info 
        << journal::loc(__HERE__)
        << "meshing body@" <<  body 
        << journal::newline;

    box bbox = get_body_box(body);
    double maxSurfaceDeviation = (bbox.high() - bbox.low()).len() / 50.0;
    info 
        << journal::loc(__HERE__)
        << "maximum surface deviation: " <<  maxSurfaceDeviation
        << journal::newline;

    PyObject * value;

    value = PyObject_GetAttrString(py_prop, "gridAspectRatio");
    if (!value) {
        return 0;
    }
    double aspectRatio = PyFloat_AsDouble(value);
    info 
        << journal::loc(__HERE__)
        << "grid aspect ratio: " << aspectRatio
        << journal::newline;


    value = PyObject_GetAttrString(py_prop, "maximumEdgeLength");
    if (!value) {
        return 0;
    }
    double maxEdgeLength = PyFloat_AsDouble(value);
    info 
        << journal::loc(__HERE__)
        << "maximum edge length: " << maxEdgeLength
        << journal::newline;


    value = PyObject_GetAttrString(py_prop, "maximumSurfaceTolerance");
    if (!value) {
        return 0;
    }
    double maxSurfaceTolerance = PyFloat_AsDouble(value);
    info 
        << journal::loc(__HERE__)
        << "maximum surface tolerance: " << maxSurfaceTolerance
        << journal::newline;


    REFINEMENT * ref = new REFINEMENT;
    ref->set_grid_aspect_ratio(aspectRatio);
    ref->set_max_edge_length(maxEdgeLength);
    ref->set_surface_tol(maxSurfaceTolerance);

    ref->set_surf_mode(AF_SURF_ALL);
    ref->set_adjust_mode(AF_ADJUST_ALL);
    ref->set_grid_mode(AF_GRID_TO_EDGES);
    ref->set_triang_mode(AF_TRIANG_ALL);

    BODY * meshed;
    outcome check = api_change_body_trans(body, NULL);
    if (!check.ok()) {
        throwACISError(check, "mesh", pyacis_runtimeError);
        return 0;
    }

    check = api_make_tri_mesh_body(body, ref, meshed);
    if (!check.ok()) {
        throwACISError(check, "mesh: api_make_tri_mesh_body", pyacis_runtimeError);
        return 0;
    }

    // return
    return PyCObject_FromVoidPtr(meshed, 0);
}


// version
// $Id: meshing.cc,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $

// End of file
