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
#include "stokes_solver.h"

extern "C" {

#include "mpi.h"
#include "global_defs.h"
#include "citcom_init.h"
#include "drive_solvers.h"

    void assemble_forces(struct All_variables*, int);
    void construct_stiffness_B_matrix(struct All_variables*);
    void get_system_viscosity(struct All_variables*, int, float**, float**);
    void set_cg_defaults(struct All_variables*);
    void set_mg_defaults(struct All_variables*);
    void solve_constrained_flow_iterative(struct All_variables*);


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



char pyRegional_general_stokes_solver_Unorm__doc__[] = "";
char pyRegional_general_stokes_solver_Unorm__name__[] = "general_stokes_solver_Unorm";

PyObject * pyRegional_general_stokes_solver_Unorm(PyObject *self, PyObject *args)
{
    double Udot_mag, dUdot_mag;

    general_stokes_solver_Unorm(E, &Udot_mag, &dUdot_mag);

    return Py_BuildValue("dd", Udot_mag, dUdot_mag);
}



char pyRegional_general_stokes_solver_fini__doc__[] = "";
char pyRegional_general_stokes_solver_fini__name__[] = "general_stokes_solver_fini";

PyObject * pyRegional_general_stokes_solver_fini(PyObject *self, PyObject *args)
{
    general_stokes_solver_fini(E);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyRegional_general_stokes_solver_init__doc__[] = "";
char pyRegional_general_stokes_solver_init__name__[] = "general_stokes_solver_init";

PyObject * pyRegional_general_stokes_solver_init(PyObject *self, PyObject *args)
{
    general_stokes_solver_init(E);

    Py_INCREF(Py_None);
    return Py_None;
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



char pyRegional_general_stokes_solver_update_velo__doc__[] = "";
char pyRegional_general_stokes_solver_update_velo__name__[] = "general_stokes_solver_update_velo";

PyObject * pyRegional_general_stokes_solver_update_velo(PyObject *self, PyObject *args)
{
    general_stokes_solver_update_velo(E);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyRegional_get_system_viscosity__doc__[] = "";
char pyRegional_get_system_viscosity__name__[] = "get_system_viscosity";

PyObject * pyRegional_get_system_viscosity(PyObject *self, PyObject *args)
{
    get_system_viscosity(E,1,E->EVI[E->mesh.levmax],E->VI[E->mesh.levmax]);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyRegional_set_cg_defaults__doc__[] = "";
char pyRegional_set_cg_defaults__name__[] = "set_cg_defaults";

PyObject * pyRegional_set_cg_defaults(PyObject *self, PyObject *args)
{
    E->control.CONJ_GRAD = 1;
    set_cg_defaults(E);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyRegional_set_mg_defaults__doc__[] = "";
char pyRegional_set_mg_defaults__name__[] = "set_mg_defaults";

PyObject * pyRegional_set_mg_defaults(PyObject *self, PyObject *args)
{
    E->control.NMULTIGRID = 1;
    set_mg_defaults(E);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyRegional_set_mg_el_defaults__doc__[] = "";
char pyRegional_set_mg_el_defaults__name__[] = "set_mg_el_defaults";

PyObject * pyRegional_set_mg_el_defaults(PyObject *self, PyObject *args)
{
    E->control.EMULTIGRID = 1;
    set_mg_defaults(E);

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


// version
// $Id: stokes_solver.cc,v 1.1 2003/07/15 00:40:03 tan2 Exp $

// End of file
