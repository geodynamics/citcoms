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
#include "global_bbox.h"
#include "global_defs.h"
#include "initTemperature.h"
#include "Boundary.h"
#include "CitcomSource.h"
#include "Convertor.h"
#include "Interior.h"
#include "exchangers.h"

void deleteBoundary(void*);
void deleteBoundedBox(void*);
void deleteInterior(void*);
void deleteCitcomSource(void*);

using Exchanger::BoundedBox;
using Exchanger::BoundedMesh;


char PyCitcomSExchanger_createBoundary__doc__[] = "";
char PyCitcomSExchanger_createBoundary__name__[] = "createBoundary";

PyObject * PyCitcomSExchanger_createBoundary(PyObject *, PyObject *args)
{
    PyObject *obj1;

    if (!PyArg_ParseTuple(args, "O:createBoundary", &obj1))
	return NULL;

    All_variables* E = static_cast<All_variables*>
	                          (PyCObject_AsVoidPtr(obj1));

    Boundary* b = new Boundary(E);
    BoundedBox* bbox = const_cast<BoundedBox*>(&(b->bbox()));

    PyObject *cobj1 = PyCObject_FromVoidPtr(b, deleteBoundary);
    PyObject *cobj2 = PyCObject_FromVoidPtr(bbox, deleteBoundedBox);
    return Py_BuildValue("OO", cobj1, cobj2);
}


char PyCitcomSExchanger_createEmptyBoundary__doc__[] = "";
char PyCitcomSExchanger_createEmptyBoundary__name__[] = "createEmptyBoundary";

PyObject * PyCitcomSExchanger_createEmptyBoundary(PyObject *, PyObject *args)
{
    Boundary* b = new Boundary();

    PyObject *cobj = PyCObject_FromVoidPtr(b, deleteBoundary);
    return Py_BuildValue("O", cobj);
}


char PyCitcomSExchanger_createEmptyInterior__doc__[] = "";
char PyCitcomSExchanger_createEmptyInterior__name__[] = "createEmptyInterior";

PyObject * PyCitcomSExchanger_createEmptyInterior(PyObject *, PyObject *args)
{
    Interior* b = new Interior();

    PyObject *cobj = PyCObject_FromVoidPtr(b, deleteInterior);
    return Py_BuildValue("O", cobj);
}


char PyCitcomSExchanger_createGlobalBoundedBox__doc__[] = "";
char PyCitcomSExchanger_createGlobalBoundedBox__name__[] = "createGlobalBoundedBox";

PyObject * PyCitcomSExchanger_createGlobalBoundedBox(PyObject *, PyObject *args)
{
    PyObject *obj1;

    if (!PyArg_ParseTuple(args, "O:createGlobalBoundedBox", &obj1))
	return NULL;

    All_variables* E = static_cast<All_variables*>
	                          (PyCObject_AsVoidPtr(obj1));

    BoundedBox* bbox = new BoundedBox(Exchanger::DIM);

    if(E->parallel.nprocxy == 12) {
	// for CitcomS Full
	fullGlobalBoundedBox(*bbox, E);
    }
    else {
	// for CitcomS Regional
	regionalGlobalBoundedBox(*bbox, E);
    }
    bbox->print("CitcomS-GlobalBBox");

    PyObject *cobj = PyCObject_FromVoidPtr(bbox, deleteBoundedBox);
    return Py_BuildValue("O", cobj);
}


char PyCitcomSExchanger_createInterior__doc__[] = "";
char PyCitcomSExchanger_createInterior__name__[] = "createInterior";

PyObject * PyCitcomSExchanger_createInterior(PyObject *, PyObject *args)
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


char PyCitcomSExchanger_initConvertor__doc__[] = "";
char PyCitcomSExchanger_initConvertor__name__[] = "initConvertor";

PyObject * PyCitcomSExchanger_initConvertor(PyObject *, PyObject *args)
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


char PyCitcomSExchanger_initTemperature__doc__[] = "";
char PyCitcomSExchanger_initTemperature__name__[] = "initTemperature";

PyObject * PyCitcomSExchanger_initTemperature(PyObject *, PyObject *args)
{
    PyObject *obj1, *obj2;

    if (!PyArg_ParseTuple(args, "OO:initTemperature", &obj1, &obj2))
	return NULL;

    BoundedBox* bbox = static_cast<BoundedBox*>(PyCObject_AsVoidPtr(obj1));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj2));

    initTemperature(*bbox, E);

    Py_INCREF(Py_None);
    return Py_None;
}


char PyCitcomSExchanger_modifyT__doc__[] = "";
char PyCitcomSExchanger_modifyT__name__[] = "modifyT";

PyObject * PyCitcomSExchanger_modifyT(PyObject *, PyObject *args)
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


char PyCitcomSExchanger_CitcomSource_create__doc__[] = "";
char PyCitcomSExchanger_CitcomSource_create__name__[] = "CitcomSource_create";

PyObject * PyCitcomSExchanger_CitcomSource_create(PyObject *self, PyObject *args)
{
    PyObject *obj1, *obj2, *obj3, *obj4;
    int sinkRank;

    if (!PyArg_ParseTuple(args, "OiOOO:CitcomSource_create",
			  &obj1, &sinkRank,
			  &obj2, &obj3, &obj4))
        return NULL;

    mpi::Communicator* temp = static_cast<mpi::Communicator*>
	                      (PyCObject_AsVoidPtr(obj1));
    MPI_Comm comm = temp->handle();

    BoundedMesh* b = static_cast<BoundedMesh*>(PyCObject_AsVoidPtr(obj2));
    BoundedBox* bbox = static_cast<BoundedBox*>(PyCObject_AsVoidPtr(obj3));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj4));

    CitcomSource* source = new CitcomSource(comm, sinkRank, *b, *bbox, E);

    PyObject *cobj = PyCObject_FromVoidPtr(source, deleteCitcomSource);
    return Py_BuildValue("O", cobj);
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


void deleteCitcomSource(void* p)
{
    delete static_cast<CitcomSource*>(p);
}


// version
// $Id: exchangers.cc,v 1.46 2004/05/11 07:55:30 tan2 Exp $

// End of file
