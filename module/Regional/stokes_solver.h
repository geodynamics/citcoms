// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_stokes_solver_h)
#define pyCitcom_stokes_solver_h

extern char pyCitcom_assemble_forces__name__[];
extern char pyCitcom_assemble_forces__doc__[];
extern "C"
PyObject * pyCitcom_assemble_forces(PyObject *, PyObject *);


extern char pyCitcom_assemble_forces_pseudo_surf__name__[];
extern char pyCitcom_assemble_forces_pseudo_surf__doc__[];
extern "C"
PyObject * pyCitcom_assemble_forces_pseudo_surf(PyObject *, PyObject *);


extern char pyCitcom_construct_stiffness_B_matrix__name__[];
extern char pyCitcom_construct_stiffness_B_matrix__doc__[];
extern "C"
PyObject * pyCitcom_construct_stiffness_B_matrix(PyObject *, PyObject *);


extern char pyCitcom_general_stokes_solver__name__[];
extern char pyCitcom_general_stokes_solver__doc__[];
extern "C"
PyObject * pyCitcom_general_stokes_solver(PyObject *, PyObject *);


extern char pyCitcom_general_stokes_solver_pseudo_surf__name__[];
extern char pyCitcom_general_stokes_solver_pseudo_surf__doc__[];
extern "C"
PyObject * pyCitcom_general_stokes_solver_pseudo_surf(PyObject *, PyObject *);


extern char pyCitcom_general_stokes_solver_setup__name__[];
extern char pyCitcom_general_stokes_solver_setup__doc__[];
extern "C"
PyObject * pyCitcom_general_stokes_solver_setup(PyObject *, PyObject *);


extern char pyCitcom_get_system_viscosity__name__[];
extern char pyCitcom_get_system_viscosity__doc__[];
extern "C"
PyObject * pyCitcom_get_system_viscosity(PyObject *, PyObject *);


extern char pyCitcom_set_cg_defaults__name__[];
extern char pyCitcom_set_cg_defaults__doc__[];
extern "C"
PyObject * pyCitcom_set_cg_defaults(PyObject *, PyObject *);


extern char pyCitcom_set_mg_defaults__name__[];
extern char pyCitcom_set_mg_defaults__doc__[];
extern "C"
PyObject * pyCitcom_set_mg_defaults(PyObject *, PyObject *);


extern char pyCitcom_set_mg_el_defaults__name__[];
extern char pyCitcom_set_mg_el_defaults__doc__[];
extern "C"
PyObject * pyCitcom_set_mg_el_defaults(PyObject *, PyObject *);


extern char pyCitcom_solve_constrained_flow_iterative__name__[];
extern char pyCitcom_solve_constrained_flow_iterative__doc__[];
extern "C"
PyObject * pyCitcom_solve_constrained_flow_iterative(PyObject *, PyObject *);


extern char pyCitcom_solve_constrained_flow_iterative_pseudo_surf__name__[];
extern char pyCitcom_solve_constrained_flow_iterative_pseudo_surf__doc__[];
extern "C"
PyObject * pyCitcom_solve_constrained_flow_iterative_pseudo_surf(PyObject *, PyObject *);

#endif

// version
// $Id: stokes_solver.h,v 1.6 2005/01/08 03:02:18 ces74 Exp $

// End of file
