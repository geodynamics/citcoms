/*
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

#include <Python.h>


#include "exceptions.h"
#include "misc.h"

#include "mpi.h"
#include "global_defs.h"
#include "citcom_init.h"
#include "advection_diffusion.h"


void full_solver_init(struct All_variables*);
void regional_solver_init(struct All_variables*);

double return1_test();
void read_instructions(struct All_variables*, char*);
double CPU_time0();

void global_default_values(struct All_variables*);
void parallel_process_termination();
void read_mat_from_file(struct All_variables*);
void read_temperature_boundary_from_file(struct All_variables*);
void read_velocity_boundary_from_file(struct All_variables*);
void set_signal();
void check_settings_consistency(struct All_variables *);
void tracer_advection(struct All_variables*);
void velocities_conform_bcs(struct All_variables*, double **);


#include "mpi/pympi.h"

/* copyright */

char pyCitcom_copyright__doc__[] = "";
char pyCitcom_copyright__name__[] = "copyright";

static char pyCitcom_copyright_note[] =
"CitcomS python module: Copyright (c) 1998-2003 California Institute of Technology";


PyObject * pyCitcom_copyright(PyObject *self, PyObject *args)
{
  return Py_BuildValue("s", pyCitcom_copyright_note);
}


char pyCitcom_return1_test__doc__[] = "";
char pyCitcom_return1_test__name__[] = "return1_test";

PyObject * pyCitcom_return1_test(PyObject *self, PyObject *args)
{
    double a;
    a = return1_test();
    return Py_BuildValue("d", a);
}


char pyCitcom_CPU_time__doc__[] = "";
char pyCitcom_CPU_time__name__[] = "CPU_time";

PyObject * pyCitcom_CPU_time(PyObject *self, PyObject *args)
{
    return Py_BuildValue("d", CPU_time0());
}


void deleteE(struct All_variables *E)
{
    free(E);
}


char pyCitcom_citcom_init__doc__[] = "";
char pyCitcom_citcom_init__name__[] = "citcom_init";

PyObject * pyCitcom_citcom_init(PyObject *self, PyObject *args)
{
    PyObject *obj, *cobj;
    struct All_variables* E;
    PyMPICommObject *pycomm;
    MPI_Comm world;

    if (!PyArg_ParseTuple(args, "O:citcom_init", &obj))
        return NULL;

    pycomm = (PyMPICommObject *)obj;
    world = pycomm->comm;

    /* Allocate global pointer E */
    E = citcom_init(&world);

    /* if E is NULL, raise an exception here. */
    if (E == NULL)
        return PyErr_Format(pyCitcom_runtimeError,
                            "%s: 'libCitcomSCommon.citcom_init' failed",
                            pyCitcom_citcom_init__name__);

    cobj = PyCObject_FromVoidPtr(E, deleteE);

    return Py_BuildValue("N", cobj);
}


char pyCitcom_citcom_finalize__doc__[] = "";
char pyCitcom_citcom_finalize__name__[] = "citcom_finalize";

PyObject * pyCitcom_citcom_finalize(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;
    int status;

    if (!PyArg_ParseTuple(args, "Oi:citcom_finalize", &obj, &status))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    citcom_finalize(E, status);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_full_solver_init__doc__[] = "";
char pyCitcom_full_solver_init__name__[] = "full_solver_init";

PyObject * pyCitcom_full_solver_init(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:full_solver_init", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    full_solver_init(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_regional_solver_init__doc__[] = "";
char pyCitcom_regional_solver_init__name__[] = "regional_solver_init";

PyObject * pyCitcom_regional_solver_init(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:regional_solver_init", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    regional_solver_init(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_global_default_values__doc__[] = "";
char pyCitcom_global_default_values__name__[] = "global_default_values";

PyObject * pyCitcom_global_default_values(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:global_default_values", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    global_default_values(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_set_signal__doc__[] = "";
char pyCitcom_set_signal__name__[] = "set_signal";

PyObject * pyCitcom_set_signal(PyObject *self, PyObject *args)
{
    set_signal();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_check_settings_consistency__doc__[] = "";
char pyCitcom_check_settings_consistency__name__[] = "check_settings_consistency";

PyObject * pyCitcom_check_settings_consistency(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:check_settings_consistency", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    check_settings_consistency(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_velocities_conform_bcs__doc__[] = "";
char pyCitcom_velocities_conform_bcs__name__[] = "velocities_conform_bcs";

PyObject * pyCitcom_velocities_conform_bcs(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:velocities_conform_bcs", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    velocities_conform_bcs(E, E->U);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_BC_update_plate_temperature__doc__[] = "";
char pyCitcom_BC_update_plate_temperature__name__[] = "BC_update_plate_temperature";

PyObject * pyCitcom_BC_update_plate_temperature(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:BC_update_plate_temperature", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    if(E->control.tbcs_file==1)
        read_temperature_boundary_from_file(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_BC_update_plate_velocity__doc__[] = "";
char pyCitcom_BC_update_plate_velocity__name__[] = "BC_update_plate_velocity";

PyObject * pyCitcom_BC_update_plate_velocity(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:BC_update_plate_velocity", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

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
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:Tracer_tracer_advection", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

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
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:Visc_update_material", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    if(E->control.mat_control==1)
      read_mat_from_file(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_return_dt__doc__[] = "";
char pyCitcom_return_dt__name__[] = "return_dt";

PyObject * pyCitcom_return_dt(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:return_dt", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    return Py_BuildValue("f", E->advection.timestep);
}


char pyCitcom_return_step__doc__[] = "";
char pyCitcom_return_step__name__[] = "return_step";

PyObject * pyCitcom_return_step(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:return_step", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    return Py_BuildValue("i", E->advection.timesteps);
}


char pyCitcom_return_t__doc__[] = "";
char pyCitcom_return_t__name__[] = "return_t";

PyObject * pyCitcom_return_t(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:return_t", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    return Py_BuildValue("f", E->monitor.elapsed_time);
}


char pyCitcom_return_rank__doc__[] = "";
char pyCitcom_return_rank__name__[] = "return_rank";

PyObject * pyCitcom_return_rank(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:return_rank", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    return Py_BuildValue("i", E->parallel.me);
}


char pyCitcom_return_pid__doc__[] = "";
char pyCitcom_return_pid__name__[] = "return_pid";

PyObject * pyCitcom_return_pid(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:return_pid", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    return Py_BuildValue("i", E->control.PID);
}


/*////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////*/



/* $Id: misc.c 14641 2009-04-08 23:38:51Z tan2 $ */

/* End of file */
