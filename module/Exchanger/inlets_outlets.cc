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

#include <portinfo>
#include <Python.h>
#include "mpi.h"
#include "mpi/Communicator.h"
#include "inlets_outlets.h"

///////////////////////////////////////////////////////////////////////////////

#include "SVTInlet.h"

extern "C" void deleteSVTInlet(void*);


char PyCitcomSExchanger_SVTInlet_create__doc__[] = "";
char PyCitcomSExchanger_SVTInlet_create__name__[] = "SVTInlet_create";

PyObject * PyCitcomSExchanger_SVTInlet_create(PyObject *self, PyObject *args)
{
    PyObject *obj1, *obj2, *obj3;

    if (!PyArg_ParseTuple(args, "OOO:SVTInlet_create",
                          &obj1, &obj2, &obj3))
        return NULL;

    Boundary* b = static_cast<Boundary*>(PyCObject_AsVoidPtr(obj1));
    Exchanger::Sink* sink = static_cast<Exchanger::Sink*>(PyCObject_AsVoidPtr(obj2));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj3));

    SVTInlet* inlet = new SVTInlet(*b, *sink, E);

    PyObject *cobj = PyCObject_FromVoidPtr(inlet, deleteSVTInlet);
    return Py_BuildValue("O", cobj);
}


void deleteSVTInlet(void* p)
{
    delete static_cast<SVTInlet*>(p);
}


///////////////////////////////////////////////////////////////////////////////

#include "TInlet.h"

extern "C" void deleteTInlet(void*);


char PyCitcomSExchanger_TInlet_create__doc__[] = "";
char PyCitcomSExchanger_TInlet_create__name__[] = "TInlet_create";

PyObject * PyCitcomSExchanger_TInlet_create(PyObject *self, PyObject *args)
{
    PyObject *obj1, *obj2, *obj3;

    if (!PyArg_ParseTuple(args, "OOO:TInlet_create",
                          &obj1, &obj2, &obj3))
        return NULL;

    Exchanger::BoundedMesh* b = static_cast<Exchanger::BoundedMesh*>(PyCObject_AsVoidPtr(obj1));
    Exchanger::Sink* sink = static_cast<Exchanger::Sink*>(PyCObject_AsVoidPtr(obj2));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj3));

    TInlet* inlet = new TInlet(*b, *sink, E);

    PyObject *cobj = PyCObject_FromVoidPtr(inlet, deleteTInlet);
    return Py_BuildValue("O", cobj);
}


void deleteTInlet(void* p)
{
    delete static_cast<TInlet*>(p);
}


///////////////////////////////////////////////////////////////////////////////

#include "SInlet.h"

extern "C" void deleteSInlet(void*);


char PyCitcomSExchanger_SInlet_create__doc__[] = "";
char PyCitcomSExchanger_SInlet_create__name__[] = "SInlet_create";

PyObject * PyCitcomSExchanger_SInlet_create(PyObject *self, PyObject *args)
{
    PyObject *obj1, *obj2, *obj3;

    if (!PyArg_ParseTuple(args, "OOO:SInlet_create",
                          &obj1, &obj2, &obj3))
        return NULL;

    Boundary* b = static_cast<Boundary*>(PyCObject_AsVoidPtr(obj1));
    Exchanger::Sink* sink = static_cast<Exchanger::Sink*>(PyCObject_AsVoidPtr(obj2));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj3));

    SInlet* inlet = new SInlet(*b, *sink, E);

    PyObject *cobj = PyCObject_FromVoidPtr(inlet, deleteSInlet);
    return Py_BuildValue("O", cobj);
}


void deleteSInlet(void* p)
{
    delete static_cast<SInlet*>(p);
}


///////////////////////////////////////////////////////////////////////////////

#include "VTInlet.h"

extern "C" void deleteVTInlet(void*);


char PyCitcomSExchanger_VTInlet_create__doc__[] = "";
char PyCitcomSExchanger_VTInlet_create__name__[] = "VTInlet_create";

PyObject * PyCitcomSExchanger_VTInlet_create(PyObject *self, PyObject *args)
{
    PyObject *obj1, *obj2, *obj3;

    if (!PyArg_ParseTuple(args, "OOO:VTInlet_create",
                          &obj1, &obj2, &obj3))
        return NULL;

    Exchanger::BoundedMesh* b = static_cast<Exchanger::BoundedMesh*>(PyCObject_AsVoidPtr(obj1));
    Exchanger::Sink* sink = static_cast<Exchanger::Sink*>(PyCObject_AsVoidPtr(obj2));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj3));

    VTInlet* inlet = new VTInlet(*b, *sink, E);

    PyObject *cobj = PyCObject_FromVoidPtr(inlet, deleteVTInlet);
    return Py_BuildValue("O", cobj);
}


void deleteVTInlet(void* p)
{
    delete static_cast<VTInlet*>(p);
}


///////////////////////////////////////////////////////////////////////////////

#include "SVTOutlet.h"

extern "C" void deleteSVTOutlet(void*);


char PyCitcomSExchanger_SVTOutlet_create__doc__[] = "";
char PyCitcomSExchanger_SVTOutlet_create__name__[] = "SVTOutlet_create";

PyObject * PyCitcomSExchanger_SVTOutlet_create(PyObject *self, PyObject *args)
{
    PyObject *obj0, *obj1;

    if (!PyArg_ParseTuple(args, "OO:SVTOutlet_create",
                          &obj0, &obj1))
        return NULL;

    CitcomSource* source = static_cast<CitcomSource*>(PyCObject_AsVoidPtr(obj0));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj1));

    SVTOutlet* outlet = new SVTOutlet(*source, E);

    PyObject *cobj = PyCObject_FromVoidPtr(outlet, deleteSVTOutlet);
    return Py_BuildValue("O", cobj);
}


void deleteSVTOutlet(void* p)
{
    delete static_cast<SVTOutlet*>(p);
}


///////////////////////////////////////////////////////////////////////////////

#include "TOutlet.h"

extern "C" void deleteTOutlet(void*);


char PyCitcomSExchanger_TOutlet_create__doc__[] = "";
char PyCitcomSExchanger_TOutlet_create__name__[] = "TOutlet_create";

PyObject * PyCitcomSExchanger_TOutlet_create(PyObject *self, PyObject *args)
{
    PyObject *obj0, *obj1;

    if (!PyArg_ParseTuple(args, "OO:TOutlet_create",
                          &obj0, &obj1))
        return NULL;

    CitcomSource* source = static_cast<CitcomSource*>(PyCObject_AsVoidPtr(obj0));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj1));

    TOutlet* outlet = new TOutlet(*source, E);

    PyObject *cobj = PyCObject_FromVoidPtr(outlet, deleteTOutlet);
    return Py_BuildValue("O", cobj);
}


void deleteTOutlet(void* p)
{
    delete static_cast<TOutlet*>(p);
}


///////////////////////////////////////////////////////////////////////////////

#include "VOutlet.h"

extern "C" void deleteVOutlet(void*);


char PyCitcomSExchanger_VOutlet_create__doc__[] = "";
char PyCitcomSExchanger_VOutlet_create__name__[] = "VOutlet_create";

PyObject * PyCitcomSExchanger_VOutlet_create(PyObject *self, PyObject *args)
{
    PyObject *obj0, *obj1;

    if (!PyArg_ParseTuple(args, "OO:VOutlet_create",
                          &obj0, &obj1))
        return NULL;

    CitcomSource* source = static_cast<CitcomSource*>(PyCObject_AsVoidPtr(obj0));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj1));

    VOutlet* outlet = new VOutlet(*source, E);

    PyObject *cobj = PyCObject_FromVoidPtr(outlet, deleteVOutlet);
    return Py_BuildValue("O", cobj);
}


void deleteVOutlet(void* p)
{
    delete static_cast<VOutlet*>(p);
}



///////////////////////////////////////////////////////////////////////////////

#include "VTOutlet.h"

extern "C" void deleteVTOutlet(void*);


char PyCitcomSExchanger_VTOutlet_create__doc__[] = "";
char PyCitcomSExchanger_VTOutlet_create__name__[] = "VTOutlet_create";

PyObject * PyCitcomSExchanger_VTOutlet_create(PyObject *self, PyObject *args)
{
    PyObject *obj0, *obj1;

    if (!PyArg_ParseTuple(args, "OO:VTOutlet_create",
                          &obj0, &obj1))
        return NULL;

    CitcomSource* source = static_cast<CitcomSource*>(PyCObject_AsVoidPtr(obj0));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj1));

    VTOutlet* outlet = new VTOutlet(*source, E);

    PyObject *cobj = PyCObject_FromVoidPtr(outlet, deleteVTOutlet);
    return Py_BuildValue("O", cobj);
}


void deleteVTOutlet(void* p)
{
    delete static_cast<VTOutlet*>(p);
}


// version
// $Id: inlets_outlets.cc 2397 2005-10-04 22:37:25Z leif $

// End of file
