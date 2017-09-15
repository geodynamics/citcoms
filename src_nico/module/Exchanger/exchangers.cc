// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//<LicenseText>
//
// CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
// Copyright (C) 2002-2005, California Institute of Technology.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//</LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include "config.h"
#include <Python.h>
#include "mpi.h"
#include "mpi/pympi.h"
#include "global_bbox.h"
#include "global_defs.h"
#include "initTemperature.h"
#include "Boundary.h"
#include "CitcomSource.h"
#include "Convertor.h"
#include "Interior.h"
#include "PInterior.h"
#include "exchangers.h"

void deleteBoundary(void*);
void deleteBoundedBox(void*);
void deleteInterior(void*);
void deletePInterior(void*);
void deleteCitcomSource(void*);

using Exchanger::BoundedBox;
using Exchanger::BoundedMesh;


char PyCitcomSExchanger_createBoundary__doc__[] = "";
char PyCitcomSExchanger_createBoundary__name__[] = "createBoundary";

PyObject * PyCitcomSExchanger_createBoundary(PyObject *, PyObject *args)
{
    PyObject *obj1;
    bool excludeTop, excludeBottom;

    if (!PyArg_ParseTuple(args, "Obb:createBoundary",
			  &obj1, &excludeTop, &excludeBottom))
	return NULL;

    All_variables* E = static_cast<All_variables*>
	                          (PyCObject_AsVoidPtr(obj1));

    Boundary* b = new Boundary(E, excludeTop, excludeBottom);
    BoundedBox* bbox = const_cast<BoundedBox*>(&(b->bbox()));

    PyObject *cobj1 = PyCObject_FromVoidPtr(b, deleteBoundary);
    /* the memory of bbox is managed by b, no need to call its d'ctor */
    PyObject *cobj2 = PyCObject_FromVoidPtr(bbox, NULL);
    return Py_BuildValue("NN", cobj1, cobj2);
}


char PyCitcomSExchanger_createEmptyBoundary__doc__[] = "";
char PyCitcomSExchanger_createEmptyBoundary__name__[] = "createEmptyBoundary";

PyObject * PyCitcomSExchanger_createEmptyBoundary(PyObject *, PyObject *args)
{
    Boundary* b = new Boundary();

    PyObject *cobj = PyCObject_FromVoidPtr(b, deleteBoundary);
    return Py_BuildValue("N", cobj);
}


char PyCitcomSExchanger_createEmptyInterior__doc__[] = "";
char PyCitcomSExchanger_createEmptyInterior__name__[] = "createEmptyInterior";

PyObject * PyCitcomSExchanger_createEmptyInterior(PyObject *, PyObject *args)
{
    Interior* b = new Interior();

    PyObject *cobj = PyCObject_FromVoidPtr(b, deleteInterior);
    return Py_BuildValue("N", cobj);
}


char PyCitcomSExchanger_createEmptyPInterior__doc__[] = "";
char PyCitcomSExchanger_createEmptyPInterior__name__[] = "createEmptyPInterior";

PyObject * PyCitcomSExchanger_createEmptyPInterior(PyObject *, PyObject *args)
{
    PInterior* b = new PInterior();

    PyObject *cobj = PyCObject_FromVoidPtr(b, deletePInterior);
    return Py_BuildValue("N", cobj);
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
    return Py_BuildValue("N", cobj);
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
    /* the memory of bbox is managed by i, no need to call its d'ctor */
    PyObject *cobj2 = PyCObject_FromVoidPtr(bbox, NULL);
    return Py_BuildValue("NN", cobj1, cobj2);
}


char PyCitcomSExchanger_createPInterior__doc__[] = "";
char PyCitcomSExchanger_createPInterior__name__[] = "createPInterior";

PyObject * PyCitcomSExchanger_createPInterior(PyObject *, PyObject *args)
{
    PyObject *obj1, *obj2;

    if (!PyArg_ParseTuple(args, "OO:createPInterior", &obj1, &obj2))
	return NULL;

    BoundedBox* rbbox = static_cast<BoundedBox*>(PyCObject_AsVoidPtr(obj1));
    All_variables* E = static_cast<All_variables*>
	                          (PyCObject_AsVoidPtr(obj2));

    PInterior* i = new PInterior(*rbbox, E);
    BoundedBox* bbox = const_cast<BoundedBox*>(&(i->bbox()));

    PyObject *cobj1 = PyCObject_FromVoidPtr(i, deletePInterior);
    /* the memory of bbox is managed by i, no need to call its d'ctor */
    PyObject *cobj2 = PyCObject_FromVoidPtr(bbox, NULL);
    return Py_BuildValue("NN", cobj1, cobj2);
}


char PyCitcomSExchanger_initConvertor__doc__[] = "";
char PyCitcomSExchanger_initConvertor__name__[] = "initConvertor";

PyObject * PyCitcomSExchanger_initConvertor(PyObject *, PyObject *args)
{
   PyObject *obj1;
   int si_unit, cartesian;

   if (!PyArg_ParseTuple(args, "iiO:initConvertor",
			 &si_unit, &cartesian, &obj1))
        return NULL;

    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj1));

    Convertor::init(si_unit, cartesian, E);

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

    PyMPICommObject* temp = (PyMPICommObject*)obj1;
    MPI_Comm comm = temp->comm;

    BoundedMesh* b = static_cast<BoundedMesh*>(PyCObject_AsVoidPtr(obj2));
    BoundedBox* bbox = static_cast<BoundedBox*>(PyCObject_AsVoidPtr(obj3));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj4));

    CitcomSource* source = new CitcomSource(comm, sinkRank, *b, *bbox, E);

    PyObject *cobj = PyCObject_FromVoidPtr(source, deleteCitcomSource);
    return Py_BuildValue("N", cobj);
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


void deletePInterior(void* p)
{
    delete static_cast<PInterior*>(p);
}


void deleteCitcomSource(void* p)
{
    delete static_cast<CitcomSource*>(p);
}


// version
// $Id: exchangers.cc 15108 2009-06-02 22:56:46Z tan2 $

// End of file
