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

#include "Boundary.h"
#include "CoarseGridExchanger.h"
#include "FineGridExchanger.h"
#include "mpi/Communicator.h"
#include "mpi/Group.h"

extern "C" {
#include "global_defs.h"
}

#include "exchangers.h"

void deleteBoundary(void*);
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

    CoarseGridExchanger *cge = new CoarseGridExchanger(
	                                comm, intercomm,
					localLeader, remoteLeader,
					E);

    PyObject *cobj = PyCObject_FromVoidPtr(cge, deleteCoarseGridExchanger);
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

    FineGridExchanger *fge = new FineGridExchanger(comm, intercomm,
						   localLeader, remoteLeader,
						   E);

    PyObject *cobj = PyCObject_FromVoidPtr(fge, deleteFineGridExchanger);
    return Py_BuildValue("O", cobj);
}



char pyExchanger_createBoundary__doc__[] = "";
char pyExchanger_createBoundary__name__[] = "createBoundary";

PyObject * pyExchanger_createBoundary(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:createBoundary", &obj))
	return NULL;

    FineGridExchanger* fge = static_cast<FineGridExchanger*>
	                                (PyCObject_AsVoidPtr(obj));

    const Boundary* b = fge->createBoundary();
    PyObject* cobj = PyCObject_FromVoidPtr((void *)b,
					   deleteBoundary);

    return Py_BuildValue("O", cobj);
}



char pyExchanger_mapBoundary__doc__[] = "";
char pyExchanger_mapBoundary__name__[] = "mapBoundary";

PyObject * pyExchanger_mapBoundary(PyObject *, PyObject *args)
{
    PyObject *obj1, *obj2;

    if (!PyArg_ParseTuple(args, "OO:mapBoundary", &obj1, &obj2))
        return NULL;

    Exchanger* pe = static_cast<Exchanger*>(PyCObject_AsVoidPtr(obj1));
    Boundary* b = static_cast<Boundary*>(PyCObject_AsVoidPtr(obj2));

    pe->mapBoundary(b);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyExchanger_receiveBoundary__doc__[] = "";
char pyExchanger_receiveBoundary__name__[] = "receiveBoundary";

PyObject * pyExchanger_receiveBoundary(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:receiveBoundary", &obj))
	return NULL;

    CoarseGridExchanger* cge = static_cast<CoarseGridExchanger*>
	                                  (PyCObject_AsVoidPtr(obj));

    const Boundary* b = cge->receiveBoundary();
    PyObject* cobj = PyCObject_FromVoidPtr((void *)b,
					   deleteBoundary);

    return Py_BuildValue("O", cobj);
}



char pyExchanger_sendBoundary__doc__[] = "";
char pyExchanger_sendBoundary__name__[] = "sendBoundary";

PyObject * pyExchanger_sendBoundary(PyObject *, PyObject *args)
{
    PyObject *obj1, *obj2;

    if (!PyArg_ParseTuple(args, "OO:sendBoundary", &obj1, &obj2))
	return NULL;

    FineGridExchanger* fge = static_cast<FineGridExchanger*>
	                                (PyCObject_AsVoidPtr(obj1));
    Boundary* b = static_cast<Boundary*>(PyCObject_AsVoidPtr(obj2));

    fge->sendBoundary(b);

    Py_INCREF(Py_None);
    return Py_None;
}


/*
char pyExchanger_rE__doc__[] = "";
char pyExchanger_rE__name__[] = "rE";

PyObject * pyExchanger_rE(PyObject *, PyObject *)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:rE", &obj))
	return NULL;

}
*/


// helper functions

void deleteBoundary(void* p) {
    std::cout << "deleting Boundary" << std::endl;
    delete static_cast<Boundary*>(p);
}



void deleteCoarseGridExchanger(void* p) {
    std::cout << "deleting CoarseGridExchanger" << std::endl;
    delete static_cast<CoarseGridExchanger*>(p);
}



void deleteFineGridExchanger(void* p) {
    std::cout << "deleting FineGridExchanger" << std::endl;
    delete static_cast<FineGridExchanger*>(p);
}



// version
// $Id: exchangers.cc,v 1.3 2003/09/09 18:25:31 tan2 Exp $

// End of file
