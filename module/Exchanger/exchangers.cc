// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <Python.h>
#include "mpi.h"
#include "mpi/Communicator.h"
#include "mpi/Group.h"
#include "global_defs.h"
#include "initTemperature.h"
#include "utilTemplate.h"
#include "utility.h"
#include "Boundary.h"
#include "BoundedBox.h"
#include "BoundaryCondition.h"
#include "DIM.h"
#include "Interior.h"
#include "InteriorImposing.h"
#include "Sink.h"
#include "Source.h"

#include "exchangers.h"

struct All_variables;

void deleteBCSink(void*);
void deleteBCSource(void*);
void deleteIISink(void*);
void deleteIISource(void*);
void deleteBoundary(void*);
void deleteBoundedBox(void*);
void deleteInterior(void*);
void deleteSink(void*);
void deleteSource(void*);

//
//


char pyExchanger_createBCSink__doc__[] = "";
char pyExchanger_createBCSink__name__[] = "createBCSink";

PyObject * pyExchanger_createBCSink(PyObject *self, PyObject *args)
{
    PyObject *obj0, *obj1, *obj2, *obj3;

    if (!PyArg_ParseTuple(args, "OOOO:createBCSink",
			  &obj0, &obj1, &obj2, &obj3))
        return NULL;

    mpi::Communicator* temp = static_cast<mpi::Communicator*>
	                      (PyCObject_AsVoidPtr(obj0));
    MPI_Comm comm = temp->handle();
    Boundary* b = static_cast<Boundary*>(PyCObject_AsVoidPtr(obj1));
    Sink* sink = static_cast<Sink*>(PyCObject_AsVoidPtr(obj2));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj3));

    BoundaryConditionSink* BCSink = new BoundaryConditionSink(comm, *b,
							      *sink, E);

    PyObject *cobj = PyCObject_FromVoidPtr(BCSink, deleteBCSink);
    return Py_BuildValue("O", cobj);
}


char pyExchanger_createBCSource__doc__[] = "";
char pyExchanger_createBCSource__name__[] = "createBCSource";

PyObject * pyExchanger_createBCSource(PyObject *self, PyObject *args)
{
    PyObject *obj1, *obj2;

    if (!PyArg_ParseTuple(args, "OO:createBCSource",
			  &obj1, &obj2))
        return NULL;

    Source* source = static_cast<Source*>(PyCObject_AsVoidPtr(obj1));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj2));

    BoundaryConditionSource* BCSource = new BoundaryConditionSource(*source, E);

    PyObject *cobj = PyCObject_FromVoidPtr(BCSource, deleteBCSource);
    return Py_BuildValue("O", cobj);
}


char pyExchanger_createIISink__doc__[] = "";
char pyExchanger_createIISink__name__[] = "createIISink";

PyObject * pyExchanger_createIISink(PyObject *self, PyObject *args)
{
    PyObject *obj1, *obj2, *obj3;

    if (!PyArg_ParseTuple(args, "OOO:createIISink",
			  &obj1, &obj2, &obj3))
        return NULL;

    Interior* b = static_cast<Interior*>(PyCObject_AsVoidPtr(obj1));
    Sink* sink = static_cast<Sink*>(PyCObject_AsVoidPtr(obj2));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj3));

    InteriorImposingSink* IISink = new InteriorImposingSink(*b, *sink, E);

    PyObject *cobj = PyCObject_FromVoidPtr(IISink, deleteIISink);
    return Py_BuildValue("O", cobj);
}


char pyExchanger_createIISource__doc__[] = "";
char pyExchanger_createIISource__name__[] = "createIISource";

PyObject * pyExchanger_createIISource(PyObject *self, PyObject *args)
{
    PyObject *obj1, *obj2;

    if (!PyArg_ParseTuple(args, "OO:createIISource",
			  &obj1, &obj2))
        return NULL;

    Source* source = static_cast<Source*>(PyCObject_AsVoidPtr(obj1));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj2));

    InteriorImposingSource* IISource = new InteriorImposingSource(*source, E);

    PyObject *cobj = PyCObject_FromVoidPtr(IISource, deleteIISource);
    return Py_BuildValue("O", cobj);
}


char pyExchanger_createBoundary__doc__[] = "";
char pyExchanger_createBoundary__name__[] = "createBoundary";

PyObject * pyExchanger_createBoundary(PyObject *, PyObject *args)
{
    PyObject *obj1, *obj2;

    if (!PyArg_ParseTuple(args, "OO:createBoundary", &obj1, &obj2))
	return NULL;

    All_variables* E = static_cast<All_variables*>
	                          (PyCObject_AsVoidPtr(obj2));

    Boundary* b = new Boundary(E);
    BoundedBox* bbox = const_cast<BoundedBox*>(&(b->bbox()));

    PyObject *cobj1 = PyCObject_FromVoidPtr(b, deleteBoundary);
    PyObject *cobj2 = PyCObject_FromVoidPtr(bbox, deleteBoundedBox);
    return Py_BuildValue("OO", cobj1, cobj2);
}


char pyExchanger_createEmptyBoundary__doc__[] = "";
char pyExchanger_createEmptyBoundary__name__[] = "createEmptyBoundary";

PyObject * pyExchanger_createEmptyBoundary(PyObject *, PyObject *args)
{
    Boundary* b = new Boundary();

    PyObject *cobj = PyCObject_FromVoidPtr(b, deleteBoundary);
    return Py_BuildValue("O", cobj);
}


char pyExchanger_createEmptyInterior__doc__[] = "";
char pyExchanger_createEmptyInterior__name__[] = "createEmptyInterior";

PyObject * pyExchanger_createEmptyInterior(PyObject *, PyObject *args)
{
    Interior* b = new Interior();

    PyObject *cobj = PyCObject_FromVoidPtr(b, deleteInterior);
    return Py_BuildValue("O", cobj);
}


char pyExchanger_createGlobalBoundedBox__doc__[] = "";
char pyExchanger_createGlobalBoundedBox__name__[] = "createGlobalBoundedBox";

PyObject * pyExchanger_createGlobalBoundedBox(PyObject *, PyObject *args)
{
    PyObject *obj1;

    if (!PyArg_ParseTuple(args, "O:createGlobalBoundedBox", &obj1))
	return NULL;

    All_variables* E = static_cast<All_variables*>
	                          (PyCObject_AsVoidPtr(obj1));

    BoundedBox* bbox = new BoundedBox(DIM);

    if(E->parallel.nprocxy == 12) {
	// for CitcomS Full
	fullGlobalBoundedBox(*bbox, E);
    }
    else {
	// for CitcomS Regional
	regionalGlobalBoundedBox(*bbox, E);
    }
    bbox->print("GlobalBBox");

    PyObject *cobj = PyCObject_FromVoidPtr(bbox, deleteBoundedBox);
    return Py_BuildValue("O", cobj);
}


char pyExchanger_createInterior__doc__[] = "";
char pyExchanger_createInterior__name__[] = "createInterior";

PyObject * pyExchanger_createInterior(PyObject *, PyObject *args)
{
    PyObject *obj1, *obj2;

    if (!PyArg_ParseTuple(args, "OO:createInterior", &obj1, &obj2))
	return NULL;

    BoundedBox* rbbox = static_cast<BoundedBox*>(PyCObject_AsVoidPtr(obj1));
    All_variables* E = static_cast<All_variables*>
	                          (PyCObject_AsVoidPtr(obj2));

    Interior* i = new Interior(*rbbox, E);
    BoundedBox* bbox = const_cast<BoundedBox*>(&(i->bbox()));

    PyObject *cobj1 = PyCObject_FromVoidPtr(i, deleteInterior);
    PyObject *cobj2 = PyCObject_FromVoidPtr(bbox, deleteBoundedBox);
    return Py_BuildValue("OO", cobj1, cobj2);
}


char pyExchanger_createSink__doc__[] = "";
char pyExchanger_createSink__name__[] = "createSink";

PyObject * pyExchanger_createSink(PyObject *self, PyObject *args)
{
    PyObject *obj1, *obj2;
    int numSrc;

    if (!PyArg_ParseTuple(args, "OiO:createSink",
			  &obj1, &numSrc, &obj2))
        return NULL;

    mpi::Communicator* temp = static_cast<mpi::Communicator*>
	                      (PyCObject_AsVoidPtr(obj1));
    MPI_Comm comm = temp->handle();

    BoundedMesh* b = static_cast<BoundedMesh*>(PyCObject_AsVoidPtr(obj2));

    Sink* sink = new Sink(comm, numSrc, *b);

    PyObject *cobj = PyCObject_FromVoidPtr(sink, deleteSink);
    return Py_BuildValue("O", cobj);
}


char pyExchanger_createSource__doc__[] = "";
char pyExchanger_createSource__name__[] = "createSource";

PyObject * pyExchanger_createSource(PyObject *self, PyObject *args)
{
    PyObject *obj1, *obj2, *obj3, *obj4;
    int sink;

    if (!PyArg_ParseTuple(args, "OiOOO:createSource",
			  &obj1, &sink,
			  &obj2, &obj3, &obj4))
        return NULL;

    mpi::Communicator* temp = static_cast<mpi::Communicator*>
	                      (PyCObject_AsVoidPtr(obj1));
    MPI_Comm comm = temp->handle();

    BoundedMesh* b = static_cast<BoundedMesh*>(PyCObject_AsVoidPtr(obj2));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj3));
    BoundedBox* bbox = static_cast<BoundedBox*>(PyCObject_AsVoidPtr(obj4));

    Source* source = new Source(comm, sink, *b, E, *bbox);

    PyObject *cobj = PyCObject_FromVoidPtr(source, deleteSource);
    return Py_BuildValue("O", cobj);
}


char pyExchanger_initTemperature__doc__[] = "";
char pyExchanger_initTemperature__name__[] = "initTemperature";

PyObject * pyExchanger_initTemperature(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:initTemperature", &obj))
	return NULL;

    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj));

    initTemperature(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyExchanger_recvTandV__doc__[] = "";
char pyExchanger_recvTandV__name__[] = "recvTandV";

PyObject * pyExchanger_recvTandV(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:recvTandV", &obj))
	return NULL;

    BoundaryConditionSink* bcs = static_cast<BoundaryConditionSink*>
	                                    (PyCObject_AsVoidPtr(obj));

    bcs->recvTandV();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyExchanger_sendTandV__doc__[] = "";
char pyExchanger_sendTandV__name__[] = "sendTandV";

PyObject * pyExchanger_sendTandV(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:sendTandV", &obj))
	return NULL;

    BoundaryConditionSource* bcs = static_cast<BoundaryConditionSource*>
	                                      (PyCObject_AsVoidPtr(obj));

    bcs->sendTandV();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyExchanger_sendTraction__doc__[] = "";
char pyExchanger_sendTraction__name__[] = "sendTraction";

PyObject * pyExchanger_sendTraction(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:sendTraction", &obj))
	return NULL;

    BoundaryConditionSource* bcs = static_cast<BoundaryConditionSource*>
	                                      (PyCObject_AsVoidPtr(obj));

    bcs->sendTraction();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyExchanger_recvT__doc__[] = "";
char pyExchanger_recvT__name__[] = "recvT";

PyObject * pyExchanger_recvT(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:recvT", &obj))
	return NULL;

    InteriorImposingSink* ics = static_cast<InteriorImposingSink*>
	                                    (PyCObject_AsVoidPtr(obj));

    ics->recvT();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyExchanger_sendT__doc__[] = "";
char pyExchanger_sendT__name__[] = "sendT";

PyObject * pyExchanger_sendT(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:sendT", &obj))
	return NULL;

    InteriorImposingSource* ics = static_cast<InteriorImposingSource*>
	                                      (PyCObject_AsVoidPtr(obj));

    ics->sendT();

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

    BoundaryConditionSink* bcs = static_cast<BoundaryConditionSink*>
	                                    (PyCObject_AsVoidPtr(obj));

    bcs->imposeBC();

    Py_INCREF(Py_None);
    return Py_None;
}

char pyExchanger_imposeIC__doc__[] = "";
char pyExchanger_imposeIC__name__[] = "imposeIC";

PyObject * pyExchanger_imposeIC(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:imposeIC", &obj))
	return NULL;

    InteriorImposingSink* ics = static_cast<InteriorImposingSink*>
	                                    (PyCObject_AsVoidPtr(obj));

    ics->imposeIC();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyExchanger_exchangeBoundedBox__doc__[] = "";
char pyExchanger_exchangeBoundedBox__name__[] = "exchangeBoundedBox";

PyObject * pyExchanger_exchangeBoundedBox(PyObject *, PyObject *args)
{
    PyObject *obj0, *obj1, *obj2;
    int target;

    if (!PyArg_ParseTuple(args, "OOOi:exchangeBoundedBox",
			  &obj0, &obj1, &obj2, &target))
	return NULL;

    BoundedBox* bbox = static_cast<BoundedBox*>(PyCObject_AsVoidPtr(obj0));

    mpi::Communicator* temp1 = static_cast<mpi::Communicator*>
  	                       (PyCObject_AsVoidPtr(obj1));
    MPI_Comm mycomm = temp1->handle();

    const int leader = 0;
    int rank;
    MPI_Comm_rank(mycomm, &rank);

    // copy contents of bbox to newbbox
    BoundedBox* newbbox = new BoundedBox(*bbox);

    if(rank == leader) {
	mpi::Communicator* temp2 = static_cast<mpi::Communicator*>
	                           (PyCObject_AsVoidPtr(obj2));
	MPI_Comm intercomm = temp2->handle();

	util::exchange(intercomm, target, *newbbox);
    }

    util::broadcast(mycomm, leader, *newbbox);
    newbbox->print("RemoteBBox");

    PyObject *cobj = PyCObject_FromVoidPtr(newbbox, deleteBoundedBox);
    return Py_BuildValue("O", cobj);
}


char pyExchanger_exchangeSignal__doc__[] = "";
char pyExchanger_exchangeSignal__name__[] = "exchangeSignal";

PyObject * pyExchanger_exchangeSignal(PyObject *, PyObject *args)
{
    int signal;
    PyObject *obj1, *obj2;
    int target;

    if (!PyArg_ParseTuple(args, "iOOi:exchangeTimestep",
			  &signal, &obj1, &obj2, &target))
	return NULL;

    mpi::Communicator* temp1 = static_cast<mpi::Communicator*>
	                       (PyCObject_AsVoidPtr(obj1));
    MPI_Comm mycomm = temp1->handle();

    const int leader = 0;
    int rank;
    MPI_Comm_rank(mycomm, &rank);

    if(rank == leader) {
	mpi::Communicator* temp2 = static_cast<mpi::Communicator*>
	                           (PyCObject_AsVoidPtr(obj2));
	MPI_Comm intercomm = temp2->handle();

	util::exchange(intercomm, target, signal);
    }

    util::broadcast(mycomm, leader, signal);

    return Py_BuildValue("i", signal);
}


char pyExchanger_exchangeTimestep__doc__[] = "";
char pyExchanger_exchangeTimestep__name__[] = "exchangeTimestep";

PyObject * pyExchanger_exchangeTimestep(PyObject *, PyObject *args)
{
    double dt;
    PyObject *obj1, *obj2;
    int target;

    if (!PyArg_ParseTuple(args, "dOOi:exchangeTimestep",
			  &dt, &obj1, &obj2, &target))
	return NULL;

    mpi::Communicator* temp1 = static_cast<mpi::Communicator*>
  	                       (PyCObject_AsVoidPtr(obj1));
    MPI_Comm mycomm = temp1->handle();

    const int leader = 0;
    int rank;
    MPI_Comm_rank(mycomm, &rank);

    if(rank == leader) {
	mpi::Communicator* temp2 = static_cast<mpi::Communicator*>
	                           (PyCObject_AsVoidPtr(obj2));
	MPI_Comm intercomm = temp2->handle();

	util::exchange(intercomm, target, dt);
    }

    util::broadcast(mycomm, leader, dt);

    return Py_BuildValue("d", dt);
}


char pyExchanger_storeTimestep__doc__[] = "";
char pyExchanger_storeTimestep__name__[] = "storeTimestep";

PyObject * pyExchanger_storeTimestep(PyObject *, PyObject *args)
{
    PyObject *obj;
    double fge_t, cge_t;

    if (!PyArg_ParseTuple(args, "Odd:storeTimestep", &obj, &fge_t, &cge_t))
	return NULL;

    BoundaryConditionSink* bcs = static_cast<BoundaryConditionSink*>
	                         (PyCObject_AsVoidPtr(obj));

    bcs->storeTimestep(fge_t, cge_t);

    Py_INCREF(Py_None);
    return Py_None;
}


// helper functions

void deleteBCSink(void* p)
{
    delete static_cast<BoundaryConditionSink*>(p);
}


void deleteBCSource(void* p)
{
    delete static_cast<BoundaryConditionSource*>(p);
}


void deleteIISink(void* p)
{
    delete static_cast<InteriorImposingSink*>(p);
}


void deleteIISource(void* p)
{
    delete static_cast<InteriorImposingSource*>(p);
}


void deleteBoundary(void* p)
{
    delete static_cast<Boundary*>(p);
}


void deleteBoundedBox(void* p)
{
    delete static_cast<BoundedBox*>(p);
}


void deleteInterior(void* p)
{
    delete static_cast<Interior*>(p);
}


void deleteSink(void* p)
{
    delete static_cast<Sink*>(p);
}


void deleteSource(void* p)
{
    delete static_cast<Source*>(p);
}



// version
// $Id: exchangers.cc,v 1.28 2003/11/23 19:06:44 ces74 Exp $

// End of file
