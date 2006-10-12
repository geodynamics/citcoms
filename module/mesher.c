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

#include <portinfo>
#include <Python.h>

#include "exceptions.h"
#include "mesher.h"

#include "global_defs.h"
#include "parallel_related.h"


void allocate_common_vars(struct All_variables*);
void allocate_velocity_vars(struct All_variables*);
void check_bc_consistency(struct All_variables*);
void construct_id(struct All_variables*);
void construct_ien(struct All_variables*);
void construct_lm(struct All_variables*);
void construct_masks(struct All_variables*);
void construct_mat_group(struct All_variables*);
void construct_shape_functions(struct All_variables*);
void construct_sub_element(struct All_variables*);
void construct_surf_det (struct All_variables*);
void construct_bdry_det (struct All_variables*);
void construct_surface (struct All_variables*);
void get_initial_elapsed_time(struct All_variables*);
void lith_age_init(struct All_variables *E);
void mass_matrix(struct All_variables*);
void output_init(struct All_variables*);
void read_mat_from_file(struct All_variables*);
void set_elapsed_time(struct All_variables*);
void set_sphere_harmonics (struct All_variables*);
void set_starting_age(struct All_variables*);
void tracer_initial_settings(struct All_variables*);
double CPU_time0();




void sphere_launch(struct All_variables *E)
    /* copied from read_instructions() */
{

    E->monitor.cpu_time_at_last_cycle =
        E->monitor.cpu_time_at_start = CPU_time0();

    output_init(E);
    (E->problem_derived_values)(E);   /* call this before global_derived_  */
    (E->solver.global_derived_values)(E);

    (E->solver.parallel_processor_setup)(E);   /* get # of proc in x,y,z */
    (E->solver.parallel_domain_decomp0)(E);  /* get local nel, nno, elx, nox et al */

    allocate_common_vars(E);
    (E->problem_allocate_vars)(E);
    (E->solver_allocate_vars)(E);

           /* logical domain */
    construct_ien(E);
    construct_surface(E);
    (E->solver.construct_boundary)(E);
    (E->solver.parallel_domain_boundary_nodes)(E);

           /* physical domain */
    (E->solver.node_locations)(E);

    if(E->control.tracer==1) {
	tracer_initial_settings(E);
	(E->problem_tracer_setup)(E);
    }

    allocate_velocity_vars(E);

    get_initial_elapsed_time(E);  /* Get elapsed time from restart run*/
    set_starting_age(E);  /* set the starting age to elapsed time, if desired */
    set_elapsed_time(E);         /* reset to elapsed time to zero, if desired */

    if(E->control.lith_age)
        lith_age_init(E);

    (E->problem_boundary_conds)(E);

    check_bc_consistency(E);

    construct_masks(E);		/* order is important here */
    construct_id(E);
    construct_lm(E);

    (E->solver.parallel_communication_routs_v)(E);
    (E->solver.parallel_communication_routs_s)(E);

    construct_sub_element(E);
    construct_shape_functions(E);

    mass_matrix(E);

    construct_surf_det (E);
    construct_bdry_det (E);

    set_sphere_harmonics (E);

    if(E->control.mat_control)
      read_mat_from_file(E);
    else
      construct_mat_group(E);

    return;
}



char pyCitcom_full_sphere_launch__doc__[] = "";
char pyCitcom_full_sphere_launch__name__[] = "full_sphere_launch";

PyObject * pyCitcom_full_sphere_launch(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:full_sphere_launch", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    sphere_launch(E);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_regional_sphere_launch__doc__[] = "";
char pyCitcom_regional_sphere_launch__name__[] = "regional_sphere_launch";

PyObject * pyCitcom_regional_sphere_launch(PyObject *self, PyObject *args)
{
    PyObject *obj;
    struct All_variables* E;

    if (!PyArg_ParseTuple(args, "O:regional_sphere_launch", &obj))
        return NULL;

    E = (struct All_variables*)(PyCObject_AsVoidPtr(obj));

    sphere_launch(E);

    Py_INCREF(Py_None);
    return Py_None;
}



/* $Id$ */

/* End of file */
