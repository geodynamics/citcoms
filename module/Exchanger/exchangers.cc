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
#include "DIM.h"
#include "Convertor.h"
#include "Interior.h"
#include "Sink.h"
#include "TractionSource.h"
#include "VTSource.h"

#include "exchangers.h"

struct All_variables;

void deleteBoundary(void*);
void deleteBoundedBox(void*);
void deleteInterior(void*);
void deleteSink(void*);
void deleteTractionSource(void*);
void deleteVTSource(void*);

//
//


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
	bbox->print("GlobalBBox");
    }
    else {
	// for CitcomS Regional
	regionalGlobalBoundedBox(*bbox, E);
    }

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


char pyExchanger_createTractionSource__doc__[] = "";
char pyExchanger_createTractionSource__name__[] = "createTractionSource";

PyObject * pyExchanger_createTractionSource(PyObject *self, PyObject *args)
{
    PyObject *obj1, *obj2, *obj3, *obj4;
    int sink;

    if (!PyArg_ParseTuple(args, "OiOOO:createTractionSource",
			  &obj1, &sink,
			  &obj2, &obj3, &obj4))
        return NULL;

    mpi::Communicator* temp = static_cast<mpi::Communicator*>
	                      (PyCObject_AsVoidPtr(obj1));
    MPI_Comm comm = temp->handle();

    Boundary* b = static_cast<Boundary*>(PyCObject_AsVoidPtr(obj2));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj3));
    BoundedBox* bbox = static_cast<BoundedBox*>(PyCObject_AsVoidPtr(obj4));

    TractionSource* tractionsource = new TractionSource(comm, sink, *b, E, *bbox);

    PyObject *cobj = PyCObject_FromVoidPtr(tractionsource, deleteTractionSource);
    return Py_BuildValue("O", cobj);
}


char pyExchanger_VTSource_create__doc__[] = "";
char pyExchanger_VTSource_create__name__[] = "VTSource_create";

PyObject * pyExchanger_VTSource_create(PyObject *self, PyObject *args)
{
    PyObject *obj1, *obj2, *obj3, *obj4;
    int sink;

    if (!PyArg_ParseTuple(args, "OiOOO:VTSource_create",
			  &obj1, &sink,
			  &obj2, &obj3, &obj4))
        return NULL;

    mpi::Communicator* temp = static_cast<mpi::Communicator*>
	                      (PyCObject_AsVoidPtr(obj1));
    MPI_Comm comm = temp->handle();

    BoundedMesh* b = static_cast<BoundedMesh*>(PyCObject_AsVoidPtr(obj2));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj3));
    BoundedBox* bbox = static_cast<BoundedBox*>(PyCObject_AsVoidPtr(obj4));

    VTSource* source = new VTSource(comm, sink, *b, E, *bbox);

    PyObject *cobj = PyCObject_FromVoidPtr(source, deleteVTSource);
    return Py_BuildValue("O", cobj);
}


char pyExchanger_initConvertor__doc__[] = "";
char pyExchanger_initConvertor__name__[] = "initConvertor";

PyObject * pyExchanger_initConvertor(PyObject *, PyObject *args)
{
   PyObject *obj1;
   int dimensional, transformational;

   if (!PyArg_ParseTuple(args, "iiO:initConvertor",
			 &dimensional, &transformational, &obj1))
        return NULL;

    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj1));

    Convertor::init(dimensional, transformational, E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyExchanger_modifyT__doc__[] = "";
char pyExchanger_modifyT__name__[] = "modifyT";

PyObject * pyExchanger_modifyT(PyObject *, PyObject *args)
{
    PyObject *obj1, *obj2;

    if (!PyArg_ParseTuple(args, "OO:modifyT", &obj1, &obj2))
        return NULL;

    BoundedBox* bbox = static_cast<BoundedBox*>(PyCObject_AsVoidPtr(obj1));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj2));

    modifyT(*bbox, E);

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

	// convert before sending
	Convertor& convertor = Convertor::instance();
	convertor.coordinate(*newbbox);

	util::exchange(intercomm, target, *newbbox);

	// unconvert after receiving
	convertor.xcoordinate(*newbbox);
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

	Convertor& convertor = Convertor::instance();
	convertor.time(dt);

	util::exchange(intercomm, target, dt);

	convertor.xtime(dt);
    }

    util::broadcast(mycomm, leader, dt);

    return Py_BuildValue("d", dt);
}


// helper functions

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


void deleteTractionSource(void* p)
{
    delete static_cast<TractionSource*>(p);
}


void deleteVTSource(void* p)
{
    delete static_cast<VTSource*>(p);
}


// version
// $Id: exchangers.cc,v 1.45 2004/03/28 23:19:00 tan2 Exp $

// End of file
