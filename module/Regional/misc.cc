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


#include "exceptions.h"
#include "misc.h"

extern "C" {

#include "mpi.h"
#include "global_defs.h"
#include "citcom_init.h"
#include "advection_diffusion.h"

    double return1_test();
    void read_instructions(struct All_variables*, char*);
    double CPU_time0();

    void global_default_values(struct All_variables*);
    void parallel_process_termination();
    void read_mat_from_file(struct All_variables*);
    void read_velocity_boundary_from_file(struct All_variables*);
    void set_signal();
    void tracer_advection(struct All_variables*);
    void velocities_conform_bcs(struct All_variables*, double **);

}

#include "mpi/Communicator.h"
#include "mpi/Group.h"

// copyright

char pyCitcom_copyright__doc__[] = "";
char pyCitcom_copyright__name__[] = "copyright";

static char pyCitcom_copyright_note[] =
"CitcomS python module: Copyright (c) 1998-2003 California Institute of Technology";


PyObject * pyCitcom_copyright(PyObject *, PyObject *)
{
  return Py_BuildValue("s", pyCitcom_copyright_note);
}

//////////////////////////////////////////////////////////////////////////
// This section is for testing or temporatory implementation
//////////////////////////////////////////////////////////////////////////



char pyCitcom_return1_test__doc__[] = "";
char pyCitcom_return1_test__name__[] = "return1_test";

PyObject * pyCitcom_return1_test(PyObject *, PyObject *)
{
    double a;
    a = return1_test();
    return Py_BuildValue("d", a);
}


char pyCitcom_read_instructions__doc__[] = "";
char pyCitcom_read_instructions__name__[] = "read_instructions";

PyObject * pyCitcom_read_instructions(PyObject *self, PyObject *args)
{
    PyObject *obj;
    char *filename;

    if (!PyArg_ParseTuple(args, "Os:read_instructions", &obj, &filename))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    read_instructions(E, filename);

    // test
    fprintf(stderr,"output file prefix: %s\n", E->control.data_file);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_CPU_time__doc__[] = "";
char pyCitcom_CPU_time__name__[] = "CPU_time";

PyObject * pyCitcom_CPU_time(PyObject *, PyObject *)
{
    return Py_BuildValue("d", CPU_time0());
}


//////////////////////////////////////////////////////////////////////////
// This section is for finished implementation
//////////////////////////////////////////////////////////////////////////

char pyCitcom_citcom_init__doc__[] = "";
char pyCitcom_citcom_init__name__[] = "citcom_init";

PyObject * pyCitcom_citcom_init(PyObject *self, PyObject *args)
{
    PyObject *Obj;

    if (!PyArg_ParseTuple(args, "O:citcom_init", &Obj))
        return NULL;

    mpi::Communicator * comm = (mpi::Communicator *) PyCObject_AsVoidPtr(Obj);
    MPI_Comm world = comm->handle();

    // Allocate global pointer E
    struct All_variables* E = citcom_init(&world);

    // if E is NULL, raise an exception here.
    if (E == NULL)
      return PyErr_NoMemory();

    PyObject *cobj = PyCObject_FromVoidPtr(E, NULL);

    return Py_BuildValue("O", cobj);
}


char pyCitcom_global_default_values__doc__[] = "";
char pyCitcom_global_default_values__name__[] = "global_default_values";

PyObject * pyCitcom_global_default_values(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:global_default_values", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    global_default_values(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_set_signal__doc__[] = "";
char pyCitcom_set_signal__name__[] = "set_signal";

PyObject * pyCitcom_set_signal(PyObject *, PyObject *)
{
    set_signal();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_velocities_conform_bcs__doc__[] = "";
char pyCitcom_velocities_conform_bcs__name__[] = "velocities_conform_bcs";

PyObject * pyCitcom_velocities_conform_bcs(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:velocities_conform_bcs", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    velocities_conform_bcs(E, E->U);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_BC_update_plate_velocity__doc__[] = "";
char pyCitcom_BC_update_plate_velocity__name__[] = "BC_update_plate_velocity";

PyObject * pyCitcom_BC_update_plate_velocity(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:BC_update_plate_velocity", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    if(E->control.vbcs_file==1)
      read_velocity_boundary_from_file(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_Tracer_tracer_advection__doc__[] = "";
char pyCitcom_Tracer_tracer_advection__name__[] = "Tracer_tracer_advection";

PyObject * pyCitcom_Tracer_tracer_advection(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:Tracer_tracer_advection", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    if(E->control.tracer==1)
      tracer_advection(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_Visc_update_material__doc__[] = "";
char pyCitcom_Visc_update_material__name__[] = "Visc_update_material";

PyObject * pyCitcom_Visc_update_material(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:Visc_update_material", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    if(E->control.mat_control==1)
      read_mat_from_file(E);

    Py_INCREF(Py_None);
    return Py_None;
}




//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////



// version
// $Id: misc.cc,v 1.25 2005/01/19 02:01:41 tan2 Exp $

// End of file
