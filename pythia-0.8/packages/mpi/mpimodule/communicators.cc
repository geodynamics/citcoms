// -*- C++ -*-
//
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                              Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2005 All Rights Reserved
//
// <LicenseText>
//
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <Python.h>
#include <mpi.h>

#include "journal/debug.h"

#include "communicators.h"
#include "exceptions.h"

#include "Communicator.h"

using namespace mpi;

// helpers
extern "C" void deleteCommunicator(void *);

// create a communicator (MPI_Comm_create)
char pympi_communicatorCreate__doc__[] = "";
char pympi_communicatorCreate__name__[] = "communicatorCreate";
PyObject * pympi_communicatorCreate(PyObject *, PyObject * args)
{
    PyObject * py_old;
    PyObject * py_group;

    int ok = PyArg_ParseTuple(args, "OO:communicatorCreate", &py_old, &py_group);

    if (!ok) {
        return 0;
    }

    // convert into the MPI objects
    Communicator * old = (Communicator *) PyCObject_AsVoidPtr(py_old);
    Group * group = (Group *) PyCObject_AsVoidPtr(py_group);


    // on null communicator
    if (!old || !group) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    Communicator * comm = old->communicator(*group);

    if (!comm) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    // return the new communicator
    return PyCObject_FromVoidPtr(comm, deleteCommunicator);
}


// create a cartesian communicator (MPI_Cart_create)
char pympi_communicatorCreateCartesian__doc__[] = "";
char pympi_communicatorCreateCartesian__name__[] = "communicatorCreateCartesian";
PyObject * pympi_communicatorCreateCartesian(PyObject *, PyObject * args)
{
    int reorder;
    PyObject * py_comm;
    PyObject * procSeq;
    PyObject * periodSeq;

    journal::debug_t info("mpi.cartesian");

    int ok = PyArg_ParseTuple(
        args, 
        "OiOO:communicatorCreateCartesian",
        &py_comm, &reorder, &procSeq, &periodSeq);

    if (!ok) {
        return 0;
    }

    // get the communicator
    Communicator * comm = (Communicator *) PyCObject_AsVoidPtr(py_comm);

    // compute the dimensionality of the communicator
    int size = PySequence_Size(procSeq);
    if (size != PySequence_Size(periodSeq)) {
        PyErr_SetString(PyExc_TypeError, "mismatch in size of processor and period lists");
    }

    info << journal::at(__HERE__) << "dimension = " << size << journal::newline;

    // allocate the arrays for the MPI call
    int * procs = new int[size];
    int * periods = new int[size];

    // copy the data over
    info << journal::at(__HERE__) << "axes: ";
    for (int dim = 0; dim < size; ++dim) {
        procs[dim] = PyInt_AsLong(PySequence_GetItem(procSeq, dim));
        periods[dim] = PyInt_AsLong(PySequence_GetItem(periodSeq, dim));
        info << " (" << procs[dim] << "," << periods[dim] << ")";
    }
    info << journal::endl;

    // make the MPI call
    Communicator * cartesian = comm->cartesian(size, procs, periods, reorder);
    info
        << journal::at(__HERE__)
        << "created cartesian@" << cartesian << " from comm@" << comm
        << journal::endl;


// clean up and return
    delete [] procs;
    delete [] periods;

    if (!cartesian) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    // return the new communicator
    return PyCObject_FromVoidPtr(cartesian, deleteCommunicator);
}


// return the communicator size (MPI_Comm_size)
char pympi_communicatorSize__doc__[] = "";
char pympi_communicatorSize__name__[] = "communicatorSize";
PyObject * pympi_communicatorSize(PyObject *, PyObject * args)
{
    PyObject * py_comm;

    int ok = PyArg_ParseTuple(args, "O:communicatorSize", &py_comm);

    if (!ok) {
        return 0;
    }

    // get the communicator
    Communicator * comm = (Communicator *) PyCObject_AsVoidPtr(py_comm);

    // return
    return PyInt_FromLong(comm->size());
}


// return the communicator rank (MPI_Comm_rank)
char pympi_communicatorRank__doc__[] = "";
char pympi_communicatorRank__name__[] = "communicatorRank";
PyObject * pympi_communicatorRank(PyObject *, PyObject * args)
{
    PyObject * py_comm;

    int ok = PyArg_ParseTuple(args, "O:communicatorRank", &py_comm);

    if (!ok) {
        return 0;
    }

    // get the communicator
    Communicator * comm = (Communicator *) PyCObject_AsVoidPtr(py_comm);

    // return
    return PyInt_FromLong(comm->rank());
}


// set a communicator barrier (MPI_Barrier)
char pympi_communicatorBarrier__doc__[] = "";
char pympi_communicatorBarrier__name__[] = "communicatorBarrier";
PyObject * pympi_communicatorBarrier(PyObject *, PyObject * args)
{
    PyObject * py_comm;

    int ok = PyArg_ParseTuple(args, "O:communicatorBarrier", &py_comm);

    if (!ok) {
        return 0;
    }

    // get the communicator
    Communicator * comm = (Communicator *) PyCObject_AsVoidPtr(py_comm);
    comm->barrier();

    // return
    Py_INCREF(Py_None);
    return Py_None;

}


// return the coordinates of the process in the cartesian communicator (MPI_Cart_coords)
char pympi_communicatorCartesianCoordinates__doc__[] = "";
char pympi_communicatorCartesianCoordinates__name__[] = "communicatorCartesianCoordinates";
PyObject * pympi_communicatorCartesianCoordinates(PyObject *, PyObject * args)
{
    int dim;
    int rank;
    PyObject * py_comm;

    int ok = PyArg_ParseTuple(
        args,
        "Oii:communicatorCartesianCoordinates",
        &py_comm, &rank, &dim);

    if (!ok) {
        return 0;
    }

    // get the communicator
    Communicator * cartesian = (Communicator *) PyCObject_AsVoidPtr(py_comm);

    // allocate room for the coordinates
    int * coordinates = new int[dim];
    for (int i=0; i<dim; ++i) {
        coordinates[i] = 0;
    }

    // dump
    journal::debug_t info("mpi.cartesian");
    if (info.state()) {
        int wr, ws;
        MPI_Comm_rank(MPI_COMM_WORLD, &wr);
        MPI_Comm_size(MPI_COMM_WORLD, &ws);
        info
            << journal::at(__HERE__)
            << "[" << wr << ":" << ws << "] "
            << "communicator@" << cartesian << ": "
            << dim << "-dim cartesian communicator, rank=" << rank
            << journal::newline;
    }

    cartesian->cartesianCoordinates(rank, dim, coordinates);
    info << "coordinates:";
    for (int i=0; i < dim; ++i) {
        info << " " << coordinates[i];
    }
    info << journal::endl;

    PyObject *value = PyTuple_New(dim);
    for (int i = 0; i < dim; ++i) {
        PyTuple_SET_ITEM(value, i, PyInt_FromLong(coordinates[i]));
    }

// clean up and return
    delete [] coordinates;
    
    return value;
}


// helpers
void deleteCommunicator(void * comm)
{
    Communicator * communicator = (Communicator *) comm;
    journal::debug_t info("mpi.fini");
    info
        << journal::at(__HERE__)
        << "[" << communicator->rank() << ":" << communicator->size() << "] "
        << "deleting comm@" << communicator
        << journal::endl;

    delete communicator;

    return;
}

// version
// $Id: communicators.cc,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $

// End of file
