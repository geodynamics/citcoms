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

#include <portinfo>
#include <Python.h>

#include "bindings.h"

#include "misc.h"          // miscellaneous methods

// the method table

struct PyMethodDef pyRegional_methods[] = {

    // dummy entry for testing
    {pyRegional_copyright__name__, 
     pyRegional_copyright,
     METH_VARARGS,
     pyRegional_copyright__doc__},

    {pyRegional_return1_test__name__, 
     pyRegional_return1_test,
     METH_VARARGS, 
     pyRegional_return1_test__doc__},

    {pyRegional_CPU_time__name__, 
     pyRegional_CPU_time,
     METH_VARARGS,
     pyRegional_CPU_time__doc__},

    {pyRegional_read_instructions__name__,
     pyRegional_read_instructions,
     METH_VARARGS,
     pyRegional_read_instructions__doc__},

    {pyRegional_get_system_viscosity__name__,
     pyRegional_get_system_viscosity,
     METH_VARARGS,
     pyRegional_get_system_viscosity__doc__},

    {pyRegional_solve_constrained_flow_iterative__name__,
     pyRegional_solve_constrained_flow_iterative,
     METH_VARARGS,
     pyRegional_solve_constrained_flow_iterative__doc__},



    {pyRegional_Citcom_Init__name__,
     pyRegional_Citcom_Init,
     METH_VARARGS,
     pyRegional_Citcom_Init__doc__},

    {pyRegional_velocities_conform_bcs__name__,
     pyRegional_velocities_conform_bcs,
     METH_VARARGS,
     pyRegional_velocities_conform_bcs__doc__},

    {pyRegional_assemble_forces__name__,
     pyRegional_assemble_forces,
     METH_VARARGS,
     pyRegional_assemble_forces__doc__},

    {pyRegional_construct_stiffness_B_matrix__name__,
     pyRegional_construct_stiffness_B_matrix,
     METH_VARARGS,
     pyRegional_construct_stiffness_B_matrix__doc__},




    {pyRegional_general_stokes_solver_init__name__,
     pyRegional_general_stokes_solver_init,
     METH_VARARGS, 
     pyRegional_general_stokes_solver_init__doc__},

    {pyRegional_general_stokes_solver_fini__name__, 
     pyRegional_general_stokes_solver_fini,
     METH_VARARGS, 
     pyRegional_general_stokes_solver_fini__doc__},

    {pyRegional_general_stokes_solver_update_velo__name__, 
     pyRegional_general_stokes_solver_update_velo,
     METH_VARARGS, 
     pyRegional_general_stokes_solver_update_velo__doc__},

    {pyRegional_general_stokes_solver_Unorm__name__, 
     pyRegional_general_stokes_solver_Unorm,
     METH_VARARGS, 
     pyRegional_general_stokes_solver_Unorm__doc__},

    {pyRegional_general_stokes_solver_log__name__, 
     pyRegional_general_stokes_solver_log,
     METH_VARARGS, 
     pyRegional_general_stokes_solver_log__doc__},



// Sentinel
    {0, 0, 0, 0}
};

// version
// $Id: bindings.cc,v 1.5 2003/05/16 21:11:53 tan2 Exp $

// End of file
