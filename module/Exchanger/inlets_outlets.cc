// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <string>
#include <Python.h>
#include "mpi.h"
#include "mpi/Communicator.h"
//#include "Boundary.h"
//#include "BoundedMesh.h"
//#include "Sink.h"
#include "inlets_outlets.h"

struct All_variables;
class Boundary;
class BoundedMesh;
class Interior;
class Sink;
class VTSource;


///////////////////////////////////////////////////////////////////////////////

#include "Inlet.h"

char pyExchanger_Inlet_impose__doc__[] = "";
char pyExchanger_Inlet_impose__name__[] = "Inlet_impose";

PyObject * pyExchanger_Inlet_impose(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:Inlet_impose", &obj))
        return NULL;

    Inlet* inlet = static_cast<Inlet*>(PyCObject_AsVoidPtr(obj));

    inlet->impose();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyExchanger_Inlet_recv__doc__[] = "";
char pyExchanger_Inlet_recv__name__[] = "Inlet_recv";

PyObject * pyExchanger_Inlet_recv(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:Inlet_recv", &obj))
        return NULL;

    Inlet* inlet = static_cast<Inlet*>(PyCObject_AsVoidPtr(obj));

    inlet->recv();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyExchanger_Inlet_storeTimestep__doc__[] = "";
char pyExchanger_Inlet_storeTimestep__name__[] = "Inlet_storeTimestep";

PyObject * pyExchanger_Inlet_storeTimestep(PyObject *self, PyObject *args)
{
    PyObject *obj;
    double fge_t, cge_t;

    if (!PyArg_ParseTuple(args, "Odd:Inlet_storeTimestep",
                          &obj, &fge_t, &cge_t))
        return NULL;

    Inlet* inlet = static_cast<Inlet*>(PyCObject_AsVoidPtr(obj));

    inlet->storeTimestep(fge_t, cge_t);

    Py_INCREF(Py_None);
    return Py_None;
}


///////////////////////////////////////////////////////////////////////////////

#include "Outlet.h"

char pyExchanger_Outlet_send__doc__[] = "";
char pyExchanger_Outlet_send__name__[] = "Outlet_send";

PyObject * pyExchanger_Outlet_send(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:Outlet_send", &obj))
        return NULL;

    Outlet* outlet = static_cast<Outlet*>(PyCObject_AsVoidPtr(obj));

    outlet->send();

    Py_INCREF(Py_None);
    return Py_None;
}


///////////////////////////////////////////////////////////////////////////////

#include "BoundaryVTInlet.h"

extern "C" void deleteBoundaryVTInlet(void*);


char pyExchanger_BoundaryVTInlet_create__doc__[] = "";
char pyExchanger_BoundaryVTInlet_create__name__[] = "BoundaryVTInlet_create";

PyObject * pyExchanger_BoundaryVTInlet_create(PyObject *self, PyObject *args)
{
    PyObject *obj0, *obj1, *obj2, *obj3;
    char* mode;

    if (!PyArg_ParseTuple(args, "OOOOs:BoundaryVTInlet_create",
                          &obj0, &obj1, &obj2, &obj3, &mode))
        return NULL;

    mpi::Communicator* temp = static_cast<mpi::Communicator*>
                              (PyCObject_AsVoidPtr(obj0));
    MPI_Comm comm = temp->handle();
    Boundary* b = static_cast<Boundary*>(PyCObject_AsVoidPtr(obj1));
    Sink* sink = static_cast<Sink*>(PyCObject_AsVoidPtr(obj2));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj3));

    BoundaryVTInlet* inlet = new BoundaryVTInlet(comm, *b, *sink, E, mode);

    PyObject *cobj = PyCObject_FromVoidPtr(inlet, deleteBoundaryVTInlet);
    return Py_BuildValue("O", cobj);
}


void deleteBoundaryVTInlet(void* p)
{
    delete static_cast<BoundaryVTInlet*>(p);
}


///////////////////////////////////////////////////////////////////////////////

#include "VTInlet.h"

extern "C" void deleteVTInlet(void*);


char pyExchanger_VTInlet_create__doc__[] = "";
char pyExchanger_VTInlet_create__name__[] = "VTInlet_create";

PyObject * pyExchanger_VTInlet_create(PyObject *self, PyObject *args)
{
    PyObject *obj1, *obj2, *obj3;
    char* mode;

    if (!PyArg_ParseTuple(args, "OOOs:VTInlet_create",
                          &obj1, &obj2, &obj3, &mode))
        return NULL;

    BoundedMesh* b = static_cast<BoundedMesh*>(PyCObject_AsVoidPtr(obj1));
    Sink* sink = static_cast<Sink*>(PyCObject_AsVoidPtr(obj2));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj3));

    VTInlet* inlet = new VTInlet(*b, *sink, E, mode);

    PyObject *cobj = PyCObject_FromVoidPtr(inlet, deleteVTInlet);
    return Py_BuildValue("O", cobj);
}


void deleteVTInlet(void* p)
{
    delete static_cast<VTInlet*>(p);
}


///////////////////////////////////////////////////////////////////////////////

#include "VTOutlet.h"

extern "C" void deleteVTOutlet(void*);


char pyExchanger_VTOutlet_create__doc__[] = "";
char pyExchanger_VTOutlet_create__name__[] = "VTOutlet_create";

PyObject * pyExchanger_VTOutlet_create(PyObject *self, PyObject *args)
{
    PyObject *obj0, *obj1;
    char* mode;

    if (!PyArg_ParseTuple(args, "OOs:VTOutlet_create",
                          &obj0, &obj1, &mode))
        return NULL;

    VTSource* source = static_cast<VTSource*>(PyCObject_AsVoidPtr(obj0));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj1));

    VTOutlet* outlet = new VTOutlet(*source, E, mode);

    PyObject *cobj = PyCObject_FromVoidPtr(outlet, deleteVTOutlet);
    return Py_BuildValue("O", cobj);
}


void deleteVTOutlet(void* p)
{
    delete static_cast<VTOutlet*>(p);
}


///////////////////////////////////////////////////////////////////////////////

#if 0
#include "TractionOutlet.h"

extern "C" void deleteTractionOutlet(void*);


char pyExchanger_TractionOutlet_create__doc__[] = "";
char pyExchanger_TractionOutlet_create__name__[] = "TractionOutlet_create";

PyObject * pyExchanger_TractionOutlet_create(PyObject *self, PyObject *args)
{
    PyObject *obj0, *obj1;
    char* mode;

    if (!PyArg_ParseTuple(args, "OOs:TractionOutlet_create",
                          &obj0, &obj1, &mode))
        return NULL;

    TractionSource* source = static_cast<TractionSource*>(PyCObject_AsVoidPtr(obj0));
    All_variables* E = static_cast<All_variables*>(PyCObject_AsVoidPtr(obj1));

    TractionOutlet* outlet = new TractionOutlet(*source, E, mode);

    PyObject *cobj = PyCObject_FromVoidPtr(outlet, deleteTractionOutlet);
    return Py_BuildValue("O", cobj);
}


void deleteTractionOutlet(void* p)
{
    delete static_cast<TractionOutlet*>(p);
}
#endif

// version
// $Id: inlets_outlets.cc,v 1.3 2004/03/11 23:23:49 tan2 Exp $

// End of file
