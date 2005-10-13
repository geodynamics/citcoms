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

#include "ports.h"
#include "exceptions.h"

#include "Communicator.h"

using namespace mpi;

// send a string
char pympi_sendString__doc__[] = "";
char pympi_sendString__name__[] = "sendString";
PyObject * pympi_sendString(PyObject *, PyObject * args)
{
    int tag;
    int len;
    int peer;
    char * str;
    PyObject * py_comm;

    int ok = PyArg_ParseTuple(args, "Oiis#:sendString", &py_comm, &peer, &tag, &str, &len);

    if (!ok) {
        return 0;
    }

    // get the communicator
    Communicator * comm = (Communicator *) PyCObject_AsVoidPtr(py_comm);

    // on null communicator
    if (!comm) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    // dump arguments
    journal::debug_t info("mpi.ports");
    info
        << journal::at(__HERE__)
        << "peer={" << peer
        << "}, tag={" << tag
        << "}, string={" << str << "}@" << len
        << journal::endl;

    // send the length of the string
    int status = MPI_Send(&len, 1, MPI_INT, peer, tag, comm->handle());

    // send the data (along with the terminating null)
    status = MPI_Send(str, len+1, MPI_CHAR, peer, tag, comm->handle());

    // return
    Py_INCREF(Py_None);
    return Py_None;
}


// receive a string
char pympi_receiveString__doc__[] = "";
char pympi_receiveString__name__[] = "receiveString";
PyObject * pympi_receiveString(PyObject *, PyObject * args)
{
    int tag;
    int peer;
    PyObject * py_comm;

    int ok = PyArg_ParseTuple(args, "Oii:receiveString", &py_comm, &peer, &tag);
    if (!ok) {
        return 0;
    }

    // setup the journal channel
    journal::debug_t info("mpi.ports");

    // get the communicator
    Communicator * comm = (Communicator *) PyCObject_AsVoidPtr(py_comm);

    // on null communicator
    if (!comm) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    // receive the length
    int len;
    MPI_Status status;
    MPI_Recv(&len, 1, MPI_INT, peer, tag, comm->handle(), &status);

    // receive the data
    char * str = new char[len+1];
    MPI_Recv(str, len+1, MPI_CHAR, peer, tag, comm->handle(), &status);

    // dump message
    info
        << journal::at(__HERE__)
        << "peer={" << peer
        << "}, tag={" << tag
        << "}, string={" << str << "}@" << len
        << journal::endl;

    // build the return value
    PyObject * value = Py_BuildValue("s", str);

    // clean up
    delete [] str;

    // return
    return value;
}


// version
// $Id: ports.cc,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $

// End of file
