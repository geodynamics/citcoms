// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyRegional_stokes_solver_h)
#define pyRegional_stokes_solver_h

extern char pyRegional_assemble_forces__name__[];
extern char pyRegional_assemble_forces__doc__[];
extern "C"
PyObject * pyRegional_assemble_forces(PyObject *, PyObject *);


extern char pyRegional_construct_stiffness_B_matrix__name__[];
extern char pyRegional_construct_stiffness_B_matrix__doc__[];
extern "C"
PyObject * pyRegional_construct_stiffness_B_matrix(PyObject *, PyObject *);


extern char pyRegional_general_stokes_solver_Unorm__name__[];
extern char pyRegional_general_stokes_solver_Unorm__doc__[];
extern "C"
PyObject * pyRegional_general_stokes_solver_Unorm(PyObject *, PyObject *);


extern char pyRegional_general_stokes_solver_fini__name__[];
extern char pyRegional_general_stokes_solver_fini__doc__[];
extern "C"
PyObject * pyRegional_general_stokes_solver_fini(PyObject *, PyObject *);


extern char pyRegional_general_stokes_solver_init__name__[];
extern char pyRegional_general_stokes_solver_init__doc__[];
extern "C"
PyObject * pyRegional_general_stokes_solver_init(PyObject *, PyObject *);


extern char pyRegional_general_stokes_solver_log__name__[];
extern char pyRegional_general_stokes_solver_log__doc__[];
extern "C"
PyObject * pyRegional_general_stokes_solver_log(PyObject *, PyObject *);


extern char pyRegional_general_stokes_solver_update_velo__name__[];
extern char pyRegional_general_stokes_solver_update_velo__doc__[];
extern "C"
PyObject * pyRegional_general_stokes_solver_update_velo(PyObject *, PyObject *);


extern char pyRegional_get_system_viscosity__name__[];
extern char pyRegional_get_system_viscosity__doc__[];
extern "C"
PyObject * pyRegional_get_system_viscosity(PyObject *, PyObject *);


extern char pyRegional_set_cg_defaults__name__[];
extern char pyRegional_set_cg_defaults__doc__[];
extern "C"
PyObject * pyRegional_set_cg_defaults(PyObject *, PyObject *);


extern char pyRegional_set_mg_defaults__name__[];
extern char pyRegional_set_mg_defaults__doc__[];
extern "C"
PyObject * pyRegional_set_mg_defaults(PyObject *, PyObject *);


extern char pyRegional_set_mg_el_defaults__name__[];
extern char pyRegional_set_mg_el_defaults__doc__[];
extern "C"
PyObject * pyRegional_set_mg_el_defaults(PyObject *, PyObject *);


extern char pyRegional_solve_constrained_flow_iterative__name__[];
extern char pyRegional_solve_constrained_flow_iterative__doc__[];
extern "C"
PyObject * pyRegional_solve_constrained_flow_iterative(PyObject *, PyObject *);


#endif

// version
// $Id: stokes_solver.h,v 1.1 2003/07/15 00:40:03 tan2 Exp $

// End of file
