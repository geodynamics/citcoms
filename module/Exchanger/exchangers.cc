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


#include "exchangers.h"

void deleteBoundary(void*);
void deleteCoarseGridExchanger(void*);
void deleteFineGridExchanger(void*);


//
//


char pyExchanger_createCoarseGridExchanger__doc__[] = "";
char pyExchanger_createCoarseGridExchanger__name__[] = "createCoarseGridExchanger";

PyObject * pyExchanger_createCoarseGridExchanger(PyObject *self, PyObject *args)
{
    PyObject *obj1, *obj2, *obj3;
    int leader, localLeader, remoteLeader;

    if (!PyArg_ParseTuple(args, "OOiiiO:createCoarseGridExchanger",
			  &obj1, &obj2,
			  &leader, &localLeader, &remoteLeader,
			  &obj3))
        return NULL;

    mpi::Communicator* temp = static_cast<mpi::Communicator*>
	                      (PyCObject_AsVoidPtr(obj1));
    MPI_Comm comm = temp->handle();

    temp = static_cast<mpi::Communicator*>
	                      (PyCObject_AsVoidPtr(obj2));
    MPI_Comm intercomm = temp->handle();

    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj3));

    CoarseGridExchanger *cge = new CoarseGridExchanger(
	                                comm, intercomm,
					leader,
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
    int leader, localLeader, remoteLeader;

    if (!PyArg_ParseTuple(args, "OOiiiO:createFineGridExchanger",
			  &obj1, &obj2,
			  &leader, &localLeader, &remoteLeader,
			  &obj3))
        return NULL;

    mpi::Communicator* temp = static_cast<mpi::Communicator*>
	                      (PyCObject_AsVoidPtr(obj1));
    MPI_Comm comm = temp->handle();

    temp = static_cast<mpi::Communicator*>
	                      (PyCObject_AsVoidPtr(obj2));
    MPI_Comm intercomm = temp->handle();

    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj3));

    int lrank;
    MPI_Comm_rank(comm, &lrank);

    int rank;
    MPI_Comm_rank(intercomm, &rank);

    FineGridExchanger *fge = new FineGridExchanger(comm, intercomm,
						   leader,
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

    fge->createBoundary();

    Py_INCREF(Py_None);
    return Py_None;
}



char pyExchanger_mapBoundary__doc__[] = "";
char pyExchanger_mapBoundary__name__[] = "mapBoundary";

PyObject * pyExchanger_mapBoundary(PyObject *, PyObject *args)
{
    PyObject *obj1;

    if (!PyArg_ParseTuple(args, "O:mapBoundary", &obj1))
        return NULL;

    Exchanger* pe = static_cast<Exchanger*>(PyCObject_AsVoidPtr(obj1));

    pe->mapBoundary();

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

    cge->receiveBoundary();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyExchanger_sendBoundary__doc__[] = "";
char pyExchanger_sendBoundary__name__[] = "sendBoundary";

PyObject * pyExchanger_sendBoundary(PyObject *, PyObject *args)
{
    PyObject *obj1;

    if (!PyArg_ParseTuple(args, "O:sendBoundary", &obj1))
	return NULL;

    FineGridExchanger* fge = static_cast<FineGridExchanger*>
	                                (PyCObject_AsVoidPtr(obj1));
    fge->sendBoundary();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyExchanger_createDataArrays__doc__[] = "";
char pyExchanger_createDataArrays__name__[] = "createDataArrays";

PyObject * pyExchanger_createDataArrays(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:createBoundary", &obj))
	return NULL;

    Exchanger* pe = static_cast<Exchanger*>(PyCObject_AsVoidPtr(obj));

    pe->createDataArrays();

    Py_INCREF(Py_None);
    return Py_None;
}



char pyExchanger_deleteDataArrays__doc__[] = "";
char pyExchanger_deleteDataArrays__name__[] = "deleteDataArrays";

PyObject * pyExchanger_deleteDataArrays(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:createBoundary", &obj))
	return NULL;

    Exchanger* pe = static_cast<Exchanger*>(PyCObject_AsVoidPtr(obj));

    pe->deleteDataArrays();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyExchanger_initTemperature__doc__[] = "";
char pyExchanger_initTemperature__name__[] = "initTemperature";

PyObject * pyExchanger_initTemperature(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:initTemperature", &obj))
	return NULL;

    Exchanger* pe = static_cast<Exchanger*>(PyCObject_AsVoidPtr(obj));

    pe->initTemperature();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyExchanger_receiveTemperature__doc__[] = "";
char pyExchanger_receiveTemperature__name__[] = "receiveTemperature";

PyObject * pyExchanger_receiveTemperature(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:receiveTemperature", &obj))
	return NULL;

    Exchanger* pe = static_cast<Exchanger*>(PyCObject_AsVoidPtr(obj));

    pe->receiveTemperature();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyExchanger_sendTemperature__doc__[] = "";
char pyExchanger_sendTemperature__name__[] = "sendTemperature";

PyObject * pyExchanger_sendTemperature(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:sendTemperature", &obj))
	return NULL;

    Exchanger* pe = static_cast<Exchanger*>(PyCObject_AsVoidPtr(obj));

    pe->sendTemperature();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyExchanger_receiveVelocities__doc__[] = "";
char pyExchanger_receiveVelocities__name__[] = "receiveVelocities";

PyObject * pyExchanger_receiveVelocities(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:receiveVelocities", &obj))
	return NULL;

    Exchanger* pe = static_cast<Exchanger*>(PyCObject_AsVoidPtr(obj));

    pe->receiveVelocities();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyExchanger_sendVelocities__doc__[] = "";
char pyExchanger_sendVelocities__name__[] = "sendVelocities";

PyObject * pyExchanger_sendVelocities(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:sendVelocities", &obj))
	return NULL;

    Exchanger* pe = static_cast<Exchanger*>(PyCObject_AsVoidPtr(obj));

    pe->sendVelocities();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyExchanger_distribute__doc__[] = "";
char pyExchanger_distribute__name__[] = "distribute";

PyObject * pyExchanger_distribute(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:distribute", &obj))
	return NULL;

    Exchanger* pe = static_cast<Exchanger*>(PyCObject_AsVoidPtr(obj));

    pe->distribute();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyExchanger_gather__doc__[] = "";
char pyExchanger_gather__name__[] = "gather";

PyObject * pyExchanger_gather(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:gather", &obj))
	return NULL;

    Exchanger* pe = static_cast<Exchanger*>(PyCObject_AsVoidPtr(obj));

    pe->gather();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyExchanger_imposeBC__doc__[] = "";
char pyExchanger_imposeBC__name__[] = "imposeBC";

PyObject * pyExchanger_imposeBC(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:imposeBC", &obj))
	return NULL;

    Exchanger* pe = static_cast<Exchanger*>(PyCObject_AsVoidPtr(obj));

    pe->imposeBC();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyExchanger_setBCFlag__doc__[] = "";
char pyExchanger_setBCFlag__name__[] = "setBCFlag";

PyObject * pyExchanger_setBCFlag(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:setBCFlag", &obj))
	return NULL;

    Exchanger* pe = static_cast<Exchanger*>(PyCObject_AsVoidPtr(obj));

    pe->setBCFlag();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyExchanger_storeTimestep__doc__[] = "";
char pyExchanger_storeTimestep__name__[] = "storeTimestep";

PyObject * pyExchanger_storeTimestep(PyObject *, PyObject *args)
{
    PyObject *obj;
    double fge_t, cge_t;

    if (!PyArg_ParseTuple(args, "Odd:storeTimestep", &obj, &fge_t, &cge_t))
	return NULL;

    FineGridExchanger* fge = static_cast<FineGridExchanger*>
	                                (PyCObject_AsVoidPtr(obj));

    fge->storeTimestep(fge_t, cge_t);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyExchanger_exchangeTimestep__doc__[] = "";
char pyExchanger_exchangeTimestep__name__[] = "exchangeTimestep";

PyObject * pyExchanger_exchangeTimestep(PyObject *, PyObject *args)
{
    PyObject *obj;
    double dt;

    if (!PyArg_ParseTuple(args, "Od:exchangeTimestep", &obj, &dt))
	return NULL;

    Exchanger* pe = static_cast<Exchanger*>(PyCObject_AsVoidPtr(obj));

    double newdt = pe->exchangeTimestep(dt);

    return Py_BuildValue("d", newdt);
}


char pyExchanger_exchangeSignal__doc__[] = "";
char pyExchanger_exchangeSignal__name__[] = "exchangeSignal";

PyObject * pyExchanger_exchangeSignal(PyObject *, PyObject *args)
{
    PyObject *obj1;
    int done;

    if (!PyArg_ParseTuple(args, "Oi:exchangeSignal", &obj1, &done))
	return NULL;

    Exchanger* pe = static_cast<Exchanger*>(PyCObject_AsVoidPtr(obj1));
    int signal = pe->exchangeSignal(done);

    return Py_BuildValue("i", signal);
}



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
// $Id: exchangers.cc,v 1.20 2003/10/02 01:14:22 tan2 Exp $

// End of file
