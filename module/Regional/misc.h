// -*- C++ -*-
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2003 All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 

#if !defined(pyRegional_misc_h)
#define pyRegional_misc_h

// copyright
extern char pyRegional_copyright__name__[];
extern char pyRegional_copyright__doc__[];
extern "C"
PyObject * pyRegional_copyright(PyObject *, PyObject *);


extern char pyRegional_return1_test__name__[];
extern char pyRegional_return1_test__doc__[];
extern "C"
PyObject * pyRegional_return1_test(PyObject *, PyObject *);


extern char pyRegional_read_instructions__name__[];
extern char pyRegional_read_instructions__doc__[];
extern "C"
PyObject * pyRegional_read_instructions(PyObject *, PyObject *);


extern char pyRegional_CPU_time__name__[];
extern char pyRegional_CPU_time__doc__[];
extern "C"
PyObject * pyRegional_CPU_time(PyObject *, PyObject *);


extern char pyRegional_get_system_viscosity__name__[];
extern char pyRegional_get_system_viscosity__doc__[];
extern "C"
PyObject * pyRegional_get_system_viscosity(PyObject *, PyObject *);


extern char pyRegional_solve_constrained_flow_iterative__name__[];
extern char pyRegional_solve_constrained_flow_iterative__doc__[];
extern "C"
PyObject * pyRegional_solve_constrained_flow_iterative(PyObject *, PyObject *);


extern char pyRegional_Citcom_Init__doc__[];
extern char pyRegional_Citcom_Init__name__[];
extern "C"
PyObject * pyRegional_Citcom_Init(PyObject *, PyObject *);


extern char pyRegional_velocities_conform_bcs__name__[];
extern char pyRegional_velocities_conform_bcs__doc__[];
extern "C"
PyObject * pyRegional_velocities_conform_bcs(PyObject *, PyObject *);


extern char pyRegional_assemble_forces__name__[];
extern char pyRegional_assemble_forces__doc__[];
extern "C"
PyObject * pyRegional_assemble_forces(PyObject *, PyObject *);


extern char pyRegional_construct_stiffness_B_matrix__name__[];
extern char pyRegional_construct_stiffness_B_matrix__doc__[];
extern "C"
PyObject * pyRegional_construct_stiffness_B_matrix(PyObject *, PyObject *);


extern char pyRegional_general_stokes_solver_init__name__[];
extern char pyRegional_general_stokes_solver_init__doc__[];
extern "C"
PyObject * pyRegional_general_stokes_solver_init(PyObject *, PyObject *);


extern char pyRegional_general_stokes_solver_fini__name__[];
extern char pyRegional_general_stokes_solver_fini__doc__[];
extern "C"
PyObject * pyRegional_general_stokes_solver_fini(PyObject *, PyObject *);


extern char pyRegional_general_stokes_solver_update_velo__name__[];
extern char pyRegional_general_stokes_solver_update_velo__doc__[];
extern "C"
PyObject * pyRegional_general_stokes_solver_update_velo(PyObject *, PyObject *);


extern char pyRegional_general_stokes_solver_Unorm__name__[];
extern char pyRegional_general_stokes_solver_Unorm__doc__[];
extern "C"
PyObject * pyRegional_general_stokes_solver_Unorm(PyObject *, PyObject *);


extern char pyRegional_general_stokes_solver_log__name__[];
extern char pyRegional_general_stokes_solver_log__doc__[];
extern "C"
PyObject * pyRegional_general_stokes_solver_log(PyObject *, PyObject *);




#endif

// version
// $Id: misc.h,v 1.5 2003/05/16 21:11:53 tan2 Exp $

// End of file
