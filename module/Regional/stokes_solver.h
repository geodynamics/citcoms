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


extern char pyCitcom_construct_stiffness_B_matrix__name__[];
extern char pyCitcom_construct_stiffness_B_matrix__doc__[];
extern "C"
PyObject * pyCitcom_construct_stiffness_B_matrix(PyObject *, PyObject *);


extern char pyCitcom_general_stokes_solver_Unorm__name__[];
extern char pyCitcom_general_stokes_solver_Unorm__doc__[];
extern "C"
PyObject * pyCitcom_general_stokes_solver_Unorm(PyObject *, PyObject *);


extern char pyCitcom_general_stokes_solver_fini__name__[];
extern char pyCitcom_general_stokes_solver_fini__doc__[];
extern "C"
PyObject * pyCitcom_general_stokes_solver_fini(PyObject *, PyObject *);


extern char pyCitcom_general_stokes_solver_init__name__[];
extern char pyCitcom_general_stokes_solver_init__doc__[];
extern "C"
PyObject * pyCitcom_general_stokes_solver_init(PyObject *, PyObject *);


extern char pyCitcom_general_stokes_solver_log__name__[];
extern char pyCitcom_general_stokes_solver_log__doc__[];
extern "C"
PyObject * pyCitcom_general_stokes_solver_log(PyObject *, PyObject *);


extern char pyCitcom_general_stokes_solver_update_velo__name__[];
extern char pyCitcom_general_stokes_solver_update_velo__doc__[];
extern "C"
PyObject * pyCitcom_general_stokes_solver_update_velo(PyObject *, PyObject *);


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


#endif

// version
// $Id: stokes_solver.h,v 1.2 2003/08/01 22:53:50 tan2 Exp $

// End of file
