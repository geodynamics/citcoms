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

    void parallel_process_termination();
    void set_convection_defaults();
    void set_signal();
    void velocities_conform_bcs(struct All_variables*, double **);

}


#include "mpi/Communicator.h"
#include "mpi/Group.h"

// copyright

char pyRegional_copyright__doc__[] = "";
char pyRegional_copyright__name__[] = "copyright";

static char pyRegional_copyright_note[] =
"CitcomSRegional python module: Copyright (c) 1998-2003 California Institute of Technology";


PyObject * pyRegional_copyright(PyObject *, PyObject *)
{
  return Py_BuildValue("s", pyRegional_copyright_note);
}

//////////////////////////////////////////////////////////////////////////
// This section is for testing or temporatory implementation
//////////////////////////////////////////////////////////////////////////



char pyRegional_return1_test__doc__[] = "";
char pyRegional_return1_test__name__[] = "return1_test";

PyObject * pyRegional_return1_test(PyObject *, PyObject *)
{
    double a;
    a = return1_test();
    return Py_BuildValue("d", a);
}


char pyRegional_read_instructions__doc__[] = "";
char pyRegional_read_instructions__name__[] = "read_instructions";

PyObject * pyRegional_read_instructions(PyObject *self, PyObject *args)
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


char pyRegional_CPU_time__doc__[] = "";
char pyRegional_CPU_time__name__[] = "CPU_time";

PyObject * pyRegional_CPU_time(PyObject *, PyObject *)
{
    return Py_BuildValue("d", CPU_time0());
}


//////////////////////////////////////////////////////////////////////////
// This section is for finished implementation
//////////////////////////////////////////////////////////////////////////

char pyRegional_citcom_init__doc__[] = "";
char pyRegional_citcom_init__name__[] = "citcom_init";

PyObject * pyRegional_citcom_init(PyObject *self, PyObject *args)
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



char pyRegional_set_convection_defaults__doc__[] = "";
char pyRegional_set_convection_defaults__name__[] = "set_convection_defaults";

PyObject * pyRegional_set_convection_defaults(PyObject *, PyObject *)
{
    set_convection_defaults();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyRegional_open_info_file__doc__[] = "";
char pyRegional_open_info_file__name__[] = "open_info_file";

PyObject * pyRegional_open_info_file(PyObject *, PyObject *)
    // copied from read_instructions()
{

    if (E->control.verbose)  {
	char output_file[255];
	sprintf(output_file,"%s.info.%d",E->control.data_file,E->parallel.me);
	E->fp_out = fopen(output_file,"w");
	if (E->fp_out == NULL) {
	    char errmsg[255];
	    sprintf(errmsg,"Cannot open file '%s'\n",output_file);
	    PyErr_SetString(PyExc_IOError, errmsg);
	    return NULL;
	}
    }
    Py_INCREF(Py_None);
    return Py_None;
}


char pyRegional_set_signal__doc__[] = "";
char pyRegional_set_signal__name__[] = "set_signal";

PyObject * pyRegional_set_signal(PyObject *, PyObject *)
{
    set_signal();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyRegional_velocities_conform_bcs__doc__[] = "";
char pyRegional_velocities_conform_bcs__name__[] = "velocities_conform_bcs";

PyObject * pyRegional_velocities_conform_bcs(PyObject *self, PyObject *args)
{
    velocities_conform_bcs(E, E->U);

    Py_INCREF(Py_None);
    return Py_None;
}





//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////



// version
// $Id: misc.cc,v 1.17 2003/07/22 21:58:08 tan2 Exp $

// End of file
