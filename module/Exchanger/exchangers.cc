// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <Python.h>
#include <iostream>

#include "CoarseGridExchanger.h"
#include "FineGridExchanger.h"
#include "mpi/Communicator.h"
#include "mpi/Group.h"


#include "exchangers.h"

void deleteCoarseGridExchanger(void*);
void deleteFineGridExchanger(void*);


// return (All_variables* E)

char pyExchanger_returnE__doc__[] = "";
char pyExchanger_returnE__name__[] = "returnE";

PyObject * pyExchanger_returnE(PyObject *, PyObject *)
{
    All_variables *E = new All_variables;

    E->parallel.me = 1;

    PyObject *cobj = PyCObject_FromVoidPtr(E, NULL);
    return Py_BuildValue("O", cobj);
}

//
//


char pyExchanger_createCoarseGridExchanger__doc__[] = "";
char pyExchanger_createCoarseGridExchanger__name__[] = "createCoarseGridExchanger";

PyObject * pyExchanger_createCoarseGridExchanger(PyObject *self, PyObject *args)
{
    PyObject *obj1, *obj2, *obj3;
    int localLeader, remoteLeader;

    if (!PyArg_ParseTuple(args, "OOiiO:createCoarseGridExchanger",
			  &obj1, &obj2,
			  &localLeader, &remoteLeader,
			  &obj3))
        return NULL;

    mpi::Communicator* temp = static_cast<mpi::Communicator*>
	                      (PyCObject_AsVoidPtr(obj1));
    MPI_Comm comm = temp->handle();

    temp = static_cast<mpi::Communicator*>
	                      (PyCObject_AsVoidPtr(obj2));
    MPI_Comm intercomm = temp->handle();

    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj3));

    //int rank;
    //MPI_Comm_rank(comm, &rank);
    //std::cout << "my rank is " << rank << std::endl;
    //std::cout << "my rank in solver is " << E->parallel.me << std::endl;

    CoarseGridExchanger fge(comm, intercomm,
			  localLeader, remoteLeader,
			  E);

    PyObject *cobj = PyCObject_FromVoidPtr(&fge, deleteCoarseGridExchanger);
    return Py_BuildValue("O", cobj);
}


char pyExchanger_createFineGridExchanger__doc__[] = "";
char pyExchanger_createFineGridExchanger__name__[] = "createFineGridExchanger";

PyObject * pyExchanger_createFineGridExchanger(PyObject *self, PyObject *args)
{
    PyObject *obj1, *obj2, *obj3;
    int localLeader, remoteLeader;

    if (!PyArg_ParseTuple(args, "OOiiO:createFineGridExchanger",
			  &obj1, &obj2,
			  &localLeader, &remoteLeader,
			  &obj3))
        return NULL;

    mpi::Communicator* temp = static_cast<mpi::Communicator*>
	                      (PyCObject_AsVoidPtr(obj1));
    MPI_Comm comm = temp->handle();

    temp = static_cast<mpi::Communicator*>
	                      (PyCObject_AsVoidPtr(obj2));
    MPI_Comm intercomm = temp->handle();

    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj3));

    //int rank;
    //MPI_Comm_rank(comm, &rank);
    //std::cout << "my rank is " << rank << std::endl;
    //std::cout << "my rank in solver is " << E->parallel.me << std::endl;

    FineGridExchanger fge(comm, intercomm,
			  localLeader, remoteLeader,
			  E);

    PyObject *cobj = PyCObject_FromVoidPtr(&fge, deleteFineGridExchanger);
    return Py_BuildValue("O", cobj);
}


// helper functions

void deleteCoarseGridExchanger(void* p) {

    delete static_cast<CoarseGridExchanger*>(p);
}



void deleteFineGridExchanger(void* p) {

    delete static_cast<FineGridExchanger*>(p);
}



// version
// $Id: exchangers.cc,v 1.1 2003/09/08 21:47:27 tan2 Exp $

// End of file
