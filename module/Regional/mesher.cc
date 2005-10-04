// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//<LicenseText>
//=====================================================================
//
//                             CitcomS.py
//                 ---------------------------------
//
//                              Authors:
//            Eh Tan, Eun-seo Choi, and Pururav Thoutireddy 
//          (c) California Institute of Technology 2002-2005
//
//        By downloading and/or installing this software you have
//       agreed to the CitcomS.py-LICENSE bundled with this software.
//             Free for non-commercial academic research ONLY.
//      This program is distributed WITHOUT ANY WARRANTY whatsoever.
//
//=====================================================================
//
//  Copyright June 2005, by the California Institute of Technology.
//  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
// 
//  Any commercial use must be negotiated with the Office of Technology
//  Transfer at the California Institute of Technology. This software
//  may be subject to U.S. export control laws and regulations. By
//  accepting this software, the user agrees to comply with all
//  applicable U.S. export laws and regulations, including the
//  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
//  the Export Administration Regulations, 15 C.F.R. 730-744. User has
//  the responsibility to obtain export licenses, or other export
//  authority as may be required before exporting such information to
//  foreign countries or providing access to foreign nationals.  In no
//  event shall the California Institute of Technology be liable to any
//  party for direct, indirect, special, incidental or consequential
//  damages, including lost profits, arising out of the use of this
//  software and its documentation, even if the California Institute of
//  Technology has been advised of the possibility of such damage.
// 
//  The California Institute of Technology specifically disclaims any
//  warranties, including the implied warranties or merchantability and
//  fitness for a particular purpose. The software and documentation
//  provided hereunder is on an "as is" basis, and the California
//  Institute of Technology has no obligations to provide maintenance,
//  support, updates, enhancements or modifications.
//
//=====================================================================
//</LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <Python.h>

#include "exceptions.h"
#include "mesher.h"

#include "global_defs.h"
#include "parallel_related.h"

extern "C" {

    void allocate_common_vars(struct All_variables*);
    void allocate_velocity_vars(struct All_variables*);
    void check_bc_consistency(struct All_variables*);
    void construct_boundary(struct All_variables*);
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
    int get_process_identifier();
    void global_derived_values(struct All_variables*);
    void lith_age_init(struct All_variables *E);
    void mass_matrix(struct All_variables*);
    void node_locations(struct All_variables*);
    void open_info(struct All_variables*);
    void open_log(struct All_variables*);
    void read_mat_from_file(struct All_variables*);
    void set_elapsed_time(struct All_variables*);
    void set_sphere_harmonics (struct All_variables*);
    void set_starting_age(struct All_variables*);
    void tracer_initial_settings(struct All_variables*);

}




void sphere_launch(struct All_variables *E)
    // copied from read_instructions()
{

    E->control.PID = get_process_identifier();

    open_log(E);
    if (E->control.verbose)
      open_info(E);

    (E->problem_derived_values)(E);   /* call this before global_derived_  */
    global_derived_values(E);

    parallel_processor_setup(E);   /* get # of proc in x,y,z */
    parallel_domain_decomp0(E);  /* get local nel, nno, elx, nox et al */

    allocate_common_vars(E);
    (E->problem_allocate_vars)(E);
    (E->solver_allocate_vars)(E);

           /* logical domain */
    construct_ien(E);
    construct_surface(E);
    construct_boundary(E);
    parallel_domain_boundary_nodes(E);

           /* physical domain */
    node_locations (E);

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

    parallel_communication_routs_v(E);
    parallel_communication_routs_s(E);

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

    if (!PyArg_ParseTuple(args, "O:full_sphere_launch", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    sphere_launch(E);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_regional_sphere_launch__doc__[] = "";
char pyCitcom_regional_sphere_launch__name__[] = "regional_sphere_launch";

PyObject * pyCitcom_regional_sphere_launch(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:regional_sphere_launch", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    sphere_launch(E);

    Py_INCREF(Py_None);
    return Py_None;
}



// version
// $Id$

// End of file
