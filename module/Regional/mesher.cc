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
#include "mesher.h"

extern "C" {

#include "mpi.h"
#include "global_defs.h"
#include "citcom_init.h"

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
    void get_initial_elapsed_time(struct All_variables*);
    void global_derived_values(struct All_variables*);
    void mass_matrix(struct All_variables*);
    void node_locations(struct All_variables*);
    void parallel_communication_routs_s(struct All_variables*);
    void parallel_communication_routs_v(struct All_variables*);
    void parallel_domain_boundary_nodes(struct All_variables*);
    void parallel_domain_decomp0(struct All_variables*);
    void parallel_processor_setup(struct All_variables*);
    void read_mat_from_file(struct All_variables*);
    void set_elapsed_time(struct All_variables*);
    void set_sphere_harmonics (struct All_variables*);
    void set_starting_age(struct All_variables*);

}




char pyRegional_mesher_setup__doc__[] = "";
char pyRegional_mesher_setup__name__[] = "mesher_setup";

PyObject * pyRegional_mesher_setup(PyObject *self, PyObject *args)
{

    (E->problem_derived_values)(E);   /* call this before global_derived_  */
    global_derived_values(E);

    parallel_processor_setup(E);   /* get # of proc in x,y,z */
    parallel_domain_decomp0(E);  /* get local nel, nno, elx, nox et al */

    allocate_common_vars(E);
    (E->problem_allocate_vars)(E);
    (E->solver_allocate_vars)(E);

           /* logical domain */
    construct_ien(E);
    parallel_domain_boundary_nodes(E);

           /* physical domain */
    node_locations (E);

    allocate_velocity_vars(E);

    get_initial_elapsed_time(E);  /* Get elapsed time from restart run*/
    set_starting_age(E);  /* set the starting age to elapsed time, if desired */
    set_elapsed_time(E);         /* reset to elapsed time to zero, if desired */

    (E->problem_boundary_conds)(E);

    check_bc_consistency(E);

    construct_masks(E);		/* order is important here */
    construct_id(E);
    construct_lm(E);

    parallel_communication_routs_v(E);
    parallel_communication_routs_s(E);

    construct_sub_element(E);
    construct_shape_functions(E);

    mass_matrix(E);

    construct_surf_det (E);

    set_sphere_harmonics (E);

    if(E->control.mat_control)
      read_mat_from_file(E);
    else
      construct_mat_group(E);


    Py_INCREF(Py_None);
    return Py_None;
}




char pyRegional_set_3dsphere_defaults__doc__[] = "";
char pyRegional_set_3dsphere_defaults__name__[] = "set_3dsphere_defaults";

PyObject * pyRegional_set_3dsphere_defaults(PyObject *self, PyObject *args)
{

    E->sphere.cap[1].theta[1] = E->control.theta_min;
    E->sphere.cap[1].theta[2] = E->control.theta_max;
    E->sphere.cap[1].theta[3] = E->control.theta_max;
    E->sphere.cap[1].theta[4] = E->control.theta_min;
    E->sphere.cap[1].fi[1] = E->control.fi_min;
    E->sphere.cap[1].fi[2] = E->control.fi_min;
    E->sphere.cap[1].fi[3] = E->control.fi_max;
    E->sphere.cap[1].fi[4] = E->control.fi_max;

    Py_INCREF(Py_None);
    return Py_None;
}





// version
// $Id: mesher.cc,v 1.1 2003/07/16 21:49:50 tan2 Exp $

// End of file
