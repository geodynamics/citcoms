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
    void read_instructions(char*);
    double CPU_time0();

    void global_default_values(struct All_variables*);
    void parallel_process_termination();
    void set_signal();
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
    char *filename;

    if (!PyArg_ParseTuple(args, "s:read_instructions", &filename))
        return NULL;

    read_instructions(filename);

    // test
    // fprintf(stderr,"output file prefix: %s\n", E->control.data_file);

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
    citcom_init(&world);

    // if E is NULL, raise an exception here.
    if (E == NULL)
      return PyErr_NoMemory();


    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_global_default_values__doc__[] = "";
char pyCitcom_global_default_values__name__[] = "global_default_values";

PyObject * pyCitcom_global_default_values(PyObject *, PyObject *)
{
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
    velocities_conform_bcs(E, E->U);

    Py_INCREF(Py_None);
    return Py_None;
}





//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////



// version
// $Id: misc.cc,v 1.21 2003/08/01 22:53:50 tan2 Exp $

// End of file
