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
#include "drive_solvers.h"
#include "advection_diffusion.h"

  double return1_test();
  void read_instructions(char*);
  double CPU_time0();
  void solve_constrained_flow_iterative(struct All_variables*);
  void get_system_viscosity(struct All_variables*, int, float**, float**);

  void construct_stiffness_B_matrix(struct All_variables*);
  void velocities_conform_bcs(struct All_variables*, double **);
  void assemble_forces(struct All_variables*, int);

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


char pyRegional_get_system_viscosity__doc__[] = "";
char pyRegional_get_system_viscosity__name__[] = "get_system_viscosity";

PyObject * pyRegional_get_system_viscosity(PyObject *self, PyObject *args)
{
    get_system_viscosity(E,1,E->EVI[E->mesh.levmax],E->VI[E->mesh.levmax]);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyRegional_solve_constrained_flow_iterative__doc__[] = "";
char pyRegional_solve_constrained_flow_iterative__name__[] = "solve_constrained_flow_iterative";

PyObject * pyRegional_solve_constrained_flow_iterative(PyObject *self, PyObject *args)
{
    solve_constrained_flow_iterative(E);

    return Py_BuildValue("d", E->viscosity.sdepv_misfit);
}


//////////////////////////////////////////////////////////////////////////
// This section is for finished implementation
//////////////////////////////////////////////////////////////////////////

char pyRegional_Citcom_Init__doc__[] = "";
char pyRegional_Citcom_Init__name__[] = "Citcom_Init";

PyObject * pyRegional_Citcom_Init(PyObject *self, PyObject *args)
{
    PyObject *Obj;

    if (!PyArg_ParseTuple(args, "O:Citcom_Init", &Obj))
        return NULL;

    mpi::Communicator * comm = (mpi::Communicator *) PyCObject_AsVoidPtr(Obj);
    MPI_Comm world = comm->handle();

//     // test
//     world = MPI_COMM_WORLD;

    // Allocate global pointer E
    Citcom_Init(&world);

    // if E is NULL, raise an exception here.
    if (E == NULL)
      return PyErr_NoMemory();


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


char pyRegional_assemble_forces__doc__[] = "";
char pyRegional_assemble_forces__name__[] = "assemble_forces";

PyObject * pyRegional_assemble_forces(PyObject *self, PyObject *args)
{
    assemble_forces(E,0);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyRegional_construct_stiffness_B_matrix__doc__[] = "";
char pyRegional_construct_stiffness_B_matrix__name__[] = "construct_stiffness_B_matrix";

PyObject * pyRegional_construct_stiffness_B_matrix(PyObject *self, PyObject *args)
{
    construct_stiffness_B_matrix(E);

    Py_INCREF(Py_None);
    return Py_None;
}


//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////


char pyRegional_general_stokes_solver_init__doc__[] = "";
char pyRegional_general_stokes_solver_init__name__[] = "general_stokes_solver_init";
PyObject * pyRegional_general_stokes_solver_init(PyObject *self, PyObject *args)
{
    general_stokes_solver_init(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyRegional_general_stokes_solver_fini__doc__[] = "";
char pyRegional_general_stokes_solver_fini__name__[] = "general_stokes_solver_fini";
PyObject * pyRegional_general_stokes_solver_fini(PyObject *self, PyObject *args)
{
    general_stokes_solver_fini(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyRegional_general_stokes_solver_update_velo__doc__[] = "";
char pyRegional_general_stokes_solver_update_velo__name__[] = "general_stokes_solver_update_velo";
PyObject * pyRegional_general_stokes_solver_update_velo(PyObject *self, PyObject *args)
{
    general_stokes_solver_update_velo(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyRegional_general_stokes_solver_Unorm__doc__[] = "";
char pyRegional_general_stokes_solver_Unorm__name__[] = "general_stokes_solver_Unorm";
PyObject * pyRegional_general_stokes_solver_Unorm(PyObject *self, PyObject *args)
{
    double Udot_mag, dUdot_mag;

    general_stokes_solver_Unorm(E, &Udot_mag, &dUdot_mag);

    return Py_BuildValue("dd", Udot_mag, dUdot_mag);
}


char pyRegional_general_stokes_solver_log__doc__[] = "";
char pyRegional_general_stokes_solver_log__name__[] = "general_stokes_solver_log";
PyObject * pyRegional_general_stokes_solver_log(PyObject *self, PyObject *args)
{
    double Udot_mag, dUdot_mag;
    int count;

    if (!PyArg_ParseTuple(args, "ddi:general_stokes_solver_log", &Udot_mag, &dUdot_mag, &count))
        return NULL;

    if(E->parallel.me==0){
	fprintf(stderr,"Stress dependent viscosity: DUdot = %.4e (%.4e) for iteration %d\n",dUdot_mag,Udot_mag,count);
	fprintf(E->fp,"Stress dependent viscosity: DUdot = %.4e (%.4e) for iteration %d\n",dUdot_mag,Udot_mag,count);
	fflush(E->fp);
    }

    Py_INCREF(Py_None);
    return Py_None;
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////



// version
// $Id: misc.cc,v 1.15 2003/06/06 19:04:36 tan2 Exp $

// End of file
