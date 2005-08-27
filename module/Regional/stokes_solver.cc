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
#include "stokes_solver.h"

#include "global_defs.h"
#include "drive_solvers.h"

extern "C" {

    void assemble_forces(struct All_variables*, int);
    void construct_stiffness_B_matrix(struct All_variables*);
    void general_stokes_solver(struct All_variables *);
    void general_stokes_solver_setup(struct All_variables*);
    void get_system_viscosity(struct All_variables*, int, float**, float**);
    void set_cg_defaults(struct All_variables*);
    void set_mg_defaults(struct All_variables*);
    void solve_constrained_flow_iterative(struct All_variables*);

    void assemble_forces_pseudo_surf(struct All_variables*, int);
    void general_stokes_solver_pseudo_surf(struct All_variables *);
    void solve_constrained_flow_iterative_pseudo_surf(struct All_variables*);
}



char pyCitcom_assemble_forces__doc__[] = "";
char pyCitcom_assemble_forces__name__[] = "assemble_forces";

PyObject * pyCitcom_assemble_forces(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:assemble_forces", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    assemble_forces(E,0);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_assemble_forces_pseudo_surf__doc__[] = "";
char pyCitcom_assemble_forces_pseudo_surf__name__[] = "assemble_forces_pseudo_surf";

PyObject * pyCitcom_assemble_forces_pseudo_surf(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:assemble_forces_pseudo_surf", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    assemble_forces_pseudo_surf(E,0);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_construct_stiffness_B_matrix__doc__[] = "";
char pyCitcom_construct_stiffness_B_matrix__name__[] = "construct_stiffness_B_matrix";

PyObject * pyCitcom_construct_stiffness_B_matrix(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:construct_stiffness_B_matrix", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    construct_stiffness_B_matrix(E);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_general_stokes_solver__doc__[] = "";
char pyCitcom_general_stokes_solver__name__[] = "general_stokes_solver";

PyObject * pyCitcom_general_stokes_solver(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:general_stokes_solver", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    if(E->control.pseudo_free_surf)
	    if(E->mesh.topvbc==2)
		    general_stokes_solver_pseudo_surf(E);
	    else
		    assert(0);
    else
	    general_stokes_solver(E);



    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_general_stokes_solver_setup__doc__[] = "";
char pyCitcom_general_stokes_solver_setup__name__[] = "general_stokes_solver_setup";

PyObject * pyCitcom_general_stokes_solver_setup(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:general_stokes_solver_setup", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    general_stokes_solver_setup(E);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_get_system_viscosity__doc__[] = "";
char pyCitcom_get_system_viscosity__name__[] = "get_system_viscosity";

PyObject * pyCitcom_get_system_viscosity(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:get_system_viscosity", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    get_system_viscosity(E,1,E->EVI[E->mesh.levmax],E->VI[E->mesh.levmax]);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_set_cg_defaults__doc__[] = "";
char pyCitcom_set_cg_defaults__name__[] = "set_cg_defaults";

PyObject * pyCitcom_set_cg_defaults(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:set_cg_defaults", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    E->control.CONJ_GRAD = 1;
    set_cg_defaults(E);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_set_mg_defaults__doc__[] = "";
char pyCitcom_set_mg_defaults__name__[] = "set_mg_defaults";

PyObject * pyCitcom_set_mg_defaults(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:set_mg_defaults", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    E->control.NMULTIGRID = 1;
    set_mg_defaults(E);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_set_mg_el_defaults__doc__[] = "";
char pyCitcom_set_mg_el_defaults__name__[] = "set_mg_el_defaults";

PyObject * pyCitcom_set_mg_el_defaults(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:set_mg_el_defaults", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    E->control.EMULTIGRID = 1;
    set_mg_defaults(E);

    Py_INCREF(Py_None);
    return Py_None;
}



char pyCitcom_solve_constrained_flow_iterative__doc__[] = "";
char pyCitcom_solve_constrained_flow_iterative__name__[] = "solve_constrained_flow_iterative";

PyObject * pyCitcom_solve_constrained_flow_iterative(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:solve_constrained_flow_iterative", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    solve_constrained_flow_iterative(E);

    return Py_BuildValue("d", E->viscosity.sdepv_misfit);
}


char pyCitcom_solve_constrained_flow_iterative_pseudo_surf__doc__[] = "";
char pyCitcom_solve_constrained_flow_iterative_pseudo_surf__name__[] = "solve_constrained_flow_iterative_pseudo_surf";

PyObject * pyCitcom_solve_constrained_flow_iterative_pseudo_surf(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:solve_constrained_flow_iterative_pseudo_surf", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    solve_constrained_flow_iterative_pseudo_surf(E);

    return Py_BuildValue("d", E->viscosity.sdepv_misfit);
}

// version
// $Id: stokes_solver.cc,v 1.11 2005/06/10 02:23:20 leif Exp $

// End of file
