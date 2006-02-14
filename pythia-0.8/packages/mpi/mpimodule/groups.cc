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

#include "groups.h"
#include "exceptions.h"

#include "Group.h"

using namespace mpi;

// helpers
extern "C" void deleteGroup(void *);

// create a communicator group (MPI_Comm_group)
char pympi_groupCreate__doc__[] = "";
char pympi_groupCreate__name__[] = "groupCreate";
PyObject * pympi_groupCreate(PyObject *, PyObject * args)
{
    PyObject * py_comm;

    int ok = PyArg_ParseTuple(args, "O:groupCreate", &py_comm);

    if (!ok) {
        return 0;
    }

    // get the communicator group
    Communicator * comm = (Communicator *) PyCObject_AsVoidPtr(py_comm);

    // on null communicator
    if (!comm) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    Group * group = Group::group(*comm);

    if (!group) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    // return
    return PyCObject_FromVoidPtr(group, deleteGroup);
}


// return the communicator group size (MPI_Group_size)
char pympi_groupSize__doc__[] = "";
char pympi_groupSize__name__[] = "groupSize";
PyObject * pympi_groupSize(PyObject *, PyObject * args)
{
    PyObject * py_grp;

    int ok = PyArg_ParseTuple(args, "O:groupSize", &py_grp);

    if (!ok) {
        return 0;
    }

    // get the communicator group
    Group * group = (Group *) PyCObject_AsVoidPtr(py_grp);

    // return
    return PyInt_FromLong(group->size());
}


// return the process rank in a given communicator group (MPI_Group_rank)
char pympi_groupRank__doc__[] = "";
char pympi_groupRank__name__[] = "groupRank";
PyObject * pympi_groupRank(PyObject *, PyObject * args)
{
    PyObject * py_grp;

    int ok = PyArg_ParseTuple(args, "O:groupRank", &py_grp);

    if (!ok) {
        return 0;
    }

    // get the communicator group
    Group * group = (Group *) PyCObject_AsVoidPtr(py_grp);

    // return
    return PyInt_FromLong(group->rank());
}


// return the process rank in a given communicator group (MPI_Group_incl)
char pympi_groupInclude__doc__[] = "";
char pympi_groupInclude__name__[] = "groupInclude";
PyObject * pympi_groupInclude(PyObject *, PyObject * args)
{
    PyObject * py_grp;
    PyObject * rankSeq;

    int ok = PyArg_ParseTuple(args, "OO:groupSize", &py_grp, &rankSeq);

    if (!ok) {
        return 0;
    }

    // get the communicator group
    Group * group = (Group *) PyCObject_AsVoidPtr(py_grp);

    // check that we got a sequence as the second argument
    if (!PySequence_Check(rankSeq)) {
        PyErr_SetString(PyExc_TypeError, "second argument must be a sequence");
        return 0;
    }

    // store the ranks in a vector
    int size = PySequence_Length(rankSeq);
    int * ranks = new int[size];

    for (int i = 0; i < size; ++i) {
        ranks[i] = PyInt_AsLong(PySequence_GetItem(rankSeq, i));
    }

    // make the MPI call
    Group * newGroup = group->include(size, ranks);

    // clean up and return
    delete [] ranks;

    if (!newGroup) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    return PyCObject_FromVoidPtr(newGroup, deleteGroup);
}


// return the process rank in a given communicator group (MPI_Group_excl)
char pympi_groupExclude__doc__[] = "";
char pympi_groupExclude__name__[] = "groupExclude";
PyObject * pympi_groupExclude(PyObject *, PyObject * args)
{
    PyObject * py_grp;
    PyObject * rankSeq;

    int ok = PyArg_ParseTuple(args, "OO:groupSize", &py_grp, &rankSeq);

    if (!ok) {
        return 0;
    }

    // get the communicator group
    Group * group = (Group *) PyCObject_AsVoidPtr(py_grp);

    // check that we got a sequence as the second argument
    if (!PySequence_Check(rankSeq)) {
        PyErr_SetString(PyExc_TypeError, "second argument must be a sequence");
        return 0;
    }

    // store the ranks in a vector
    int size = PySequence_Length(rankSeq);
    int * ranks = new int[size];

    for (int i = 0; i < size; ++i) {
        ranks[i] = PyInt_AsLong(PySequence_GetItem(rankSeq, i));
    }

    // make the MPI call
    Group * newGroup = group->exclude(size, ranks);

    // clean up and return
    delete [] ranks;

    if (!newGroup) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    return PyCObject_FromVoidPtr(newGroup, deleteGroup);
}


// helpers
void deleteGroup(void * group)
{
    journal::debug_t info("mpi.fini");
    info
        << journal::at(__HERE__)
        << "group@" << group << ": deleting"
        << journal::endl;

    delete (Group *)group;

    return;
}


// version
// $Id: groups.cc,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $

// End of file
