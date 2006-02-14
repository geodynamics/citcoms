// -*- C++ -*-
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2005  All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 

#include <portinfo>

#if defined(WITH_MPI)

#include <Python.h>
#include <mpi.h>

// mesh
#include "pyre/geometry/CanonicalMesh.h"
typedef pyre::geometry::CanonicalMesh<double> mesh_t;

// debug
#include "journal/debug.h"

// my headers
#include "via_mpi.h"


// helpers
const int btag = 13;
const int ntag = 14;
const int vtag = 15;
const int ctag = 16;
const int ptag = 17;

// sendBoundary

char pyelc_sendBoundaryMPI__doc__[] = "send boundary data";
char pyelc_sendBoundaryMPI__name__[] = "sendBoundaryMPI";

PyObject * pyelc_sendBoundaryMPI(PyObject *, PyObject * args)
{
    int source, sink;
    PyObject * py_bndry;

    int ok = PyArg_ParseTuple(args, "Oii:sendBoundaryMPI", &py_bndry, &source, &sink); 

    if (!ok) {
        return 0;
    }

    mesh_t * mesh = static_cast<mesh_t *>(PyCObject_AsVoidPtr(py_bndry));

    if (PyErr_Occurred()) {
        return 0;
    }
    
    journal::debug_t info("elc.exchange");
    info 
        << journal::at(__HERE__) << "node " << source 
        << ": sending mesh@0x" << mesh << " to node " << sink 
        << journal::endl;

    // send node and facet counts
    int nodes = mesh->vertexCount();
    int facets = mesh->simplexCount();

    int counts[] = {nodes, facets};

    info 
        << journal::at(__HERE__) << "node " << source 
        << ": sending counts: " << counts[0] 
        << " nodes and " << counts[1] << " facets" << journal::endl;
    MPI_Send(counts, 2, MPI_INT, sink, btag, MPI_COMM_WORLD);
    info
        << journal::at(__HERE__) << "node " << source 
        << ": counts sent to " << sink 
        << journal::endl;

    // send the coordinates
    info 
        << journal::at(__HERE__) << "node " << source 
        << ": sending coordinates for " << nodes
        << " nodes (" << 3*nodes << " doubles)"
        << journal::endl;
    MPI_Send(mesh->vertices(), 3*nodes, MPI_DOUBLE, sink, ntag,
        MPI_COMM_WORLD);
    info 
        << journal::at(__HERE__) << "node " << source 
        << ": coordinates sent to " << sink
        << journal::endl;

    // send the connectivity
    info 
        << journal::at(__HERE__) << "node " << source 
        << ": sending connectivity for " << facets
        << " facets (" << 3*facets << " ints)"
        << journal::endl;
    MPI_Send(mesh->simplices(), 3*facets, MPI_INT, sink, ctag, MPI_COMM_WORLD);
    info
        << journal::at(__HERE__) << "node " << source 
        << ": connectivity sent to " << sink
        << journal::endl;

    // return
    Py_INCREF(Py_None);
    return Py_None;
}

// receiveBoundaryMPI

char pyelc_receiveBoundaryMPI__doc__[] = "receive boundary data";
char pyelc_receiveBoundaryMPI__name__[] = "receiveBoundaryMPI";

PyObject * pyelc_receiveBoundaryMPI(PyObject *, PyObject * args)
{
    int source, sink;
    PyObject * py_bndry;

    int ok = PyArg_ParseTuple(args, "Oii:sendBoundaryMPI", &py_bndry, &source, &sink); 

    if (!ok) {
        return 0;
    }

    mesh_t * mesh = static_cast<mesh_t *>(PyCObject_AsVoidPtr(py_bndry));

    if (PyErr_Occurred()) {
        return 0;
    }

    journal::debug_t info("elc.exchange");
    info << journal::at(__HERE__) << "node " << sink 
         << ": receiving boundary@" << mesh << " from node " << source
         << journal::endl;

    // get node and facet counts
    MPI_Status status;

    // collect the number of nodes and facets from the boundary source
    int counts[] = {0, 0};

    info 
        << journal::at(__HERE__) << "node " << sink 
        << ": waiting for the counts from " << source
         << journal::endl;
    MPI_Recv(counts, 2, MPI_INT, source, btag, MPI_COMM_WORLD, &status);
    info 
        << journal::at(__HERE__) << "node " << sink 
        << ": received counts: " << counts[0]
        << " nodes and " << counts[1] << " facets"
        << journal::endl;

    // set node and facet counts
    int nodes = counts[0];
    int facets = counts[1];

    // allocate memory
    mesh->vertexCount(nodes);
    mesh->simplexCount(facets);

    // get the coordinates
    info
        << journal::at(__HERE__) << "node " << sink 
        << ": waiting for node coordinates from "
        << source
        << journal::endl;
    MPI_Recv(mesh->vertices(), 3*nodes, MPI_DOUBLE, source, ntag, 
        MPI_COMM_WORLD, &status);
    info
        << journal::at(__HERE__) << "node " << sink 
        << ": received coordinates for " << nodes
        << " nodes (" << 3*nodes << " doubles)"
        << journal::endl;

    // get the connectivities
    info
        << journal::at(__HERE__) << "node " << sink 
        << ": waiting for node connectivity from "
        << source 
        << journal::endl;
    MPI_Recv(mesh->simplices(), 3*facets, MPI_INT, source, ctag, 
        MPI_COMM_WORLD, &status);
    info
        << journal::at(__HERE__) << "node " << sink 
        << ": received connectivity for " << facets
        << " facets (" << 3*facets << " ints)"
        << journal::endl;

    // return
    Py_INCREF(Py_None);
    return Py_None;
}

// sendFieldMPI

char pyelc_sendFieldMPI__doc__[] = "send field";
char pyelc_sendFieldMPI__name__[] = "sendFieldMPI";

PyObject * pyelc_sendFieldMPI(PyObject *, PyObject * args)
{
    int source, sink, length;
    PyObject * py_field;

    int ok = PyArg_ParseTuple(args, "iiOi:sendFieldMPI", &source, &sink, &py_field, &length); 

    if (!ok) {
        return 0;
    }
    
    double * field = static_cast<double *>(PyCObject_AsVoidPtr(py_field));

    if (PyErr_Occurred()) {
        return 0;
    }

    // send the node velocities
    journal::debug_t info("elc.exchange");
    info
        << journal::at(__HERE__) << "node " << source
        << ": sending field to " << sink
        << " (" << length << " doubles)"
        << journal::endl;
    MPI_Send(field, length, MPI_DOUBLE, sink, vtag, MPI_COMM_WORLD);
    info 
        << journal::at(__HERE__) << "node " << source
        << ": field sent to " << sink
        << journal::endl;

    // return
    Py_INCREF(Py_None);
    return Py_None;
}


// receiveFieldMPI

char pyelc_receiveFieldMPI__doc__[] = "receive boundary velocities";
char pyelc_receiveFieldMPI__name__[] = "receiveFieldMPI";

PyObject * pyelc_receiveFieldMPI(PyObject *, PyObject * args)
{
    int source, sink, length;
    PyObject * py_field;

    int ok = PyArg_ParseTuple(args, "iiOi:receiveFieldMPI", &source, &sink, &py_field, &length); 

    if (!ok) {
        return 0;
    }

    double * field = static_cast<double *>(PyCObject_AsVoidPtr(py_field));

    if (PyErr_Occurred()) {
        return 0;
    }

    MPI_Status status;
    journal::debug_t info("elc.exchange");

    // get the velocities
    info 
        << journal::at(__HERE__) << "node " << sink
        << ": waiting for field from " << source
        << journal::endl;
    MPI_Recv(field, length, MPI_DOUBLE, source, vtag, MPI_COMM_WORLD, &status);
    info 
        << journal::at(__HERE__) << "node " << sink
        << ": received field from " << source
        << " (" << length << " doubles)"
        << journal::endl;

    // return
    Py_INCREF(Py_None);
    return Py_None;
}

#endif

// $Id: via_mpi.cc,v 1.1.1.1 2005/03/08 16:13:29 aivazis Exp $

// End of file
