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

#include "bindings.h"

#include "advdiffu.h"
#include "mesher.h"
#include "misc.h"          // miscellaneous methods
#include "outputs.h"
#include "setProperties.h"
#include "stokes_solver.h"


// the method table

struct PyMethodDef pyRegional_methods[] = {

    // dummy entry for testing
    {pyRegional_copyright__name__,
     pyRegional_copyright,
     METH_VARARGS,
     pyRegional_copyright__doc__},


    //////////////////////////////////////////////////////////////////////////
    // This section is for testing or temporatory implementation
    //////////////////////////////////////////////////////////////////////////

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


    //////////////////////////////////////////////////////////////////////////
    // This section is for finished implementation
    //////////////////////////////////////////////////////////////////////////

    // from misc.h

    {pyRegional_citcom_init__name__,
     pyRegional_citcom_init,
     METH_VARARGS,
     pyRegional_citcom_init__doc__},

    {pyRegional_global_default_values__name__,
     pyRegional_global_default_values,
     METH_VARARGS,
     pyRegional_global_default_values__doc__},

    {pyRegional_set_signal__name__,
     pyRegional_set_signal,
     METH_VARARGS,
     pyRegional_set_signal__doc__},

    {pyRegional_velocities_conform_bcs__name__,
     pyRegional_velocities_conform_bcs,
     METH_VARARGS,
     pyRegional_velocities_conform_bcs__doc__},

    // from advdiffu.h

    {pyRegional_PG_timestep_init__name__,
     pyRegional_PG_timestep_init,
     METH_VARARGS,
     pyRegional_PG_timestep_init__doc__},

    {pyRegional_PG_timestep_solve__name__,
     pyRegional_PG_timestep_solve,
     METH_VARARGS,
     pyRegional_PG_timestep_solve__doc__},

    {pyRegional_set_convection_defaults__name__,
     pyRegional_set_convection_defaults,
     METH_VARARGS,
     pyRegional_set_convection_defaults__doc__},

    // from mesher.h

    {pyRegional_regional_sphere_setup__name__,
     pyRegional_regional_sphere_setup,
     METH_VARARGS,
     pyRegional_regional_sphere_setup__doc__},

    {pyRegional_regional_sphere_init__name__,
     pyRegional_regional_sphere_init,
     METH_VARARGS,
     pyRegional_regional_sphere_init__doc__},

    // from outputs.h

    {pyRegional_output__name__,
     pyRegional_output,
     METH_VARARGS,
     pyRegional_output__doc__},

    // from setProperties.h

    {pyRegional_Advection_diffusion_set_properties__name__,
     pyRegional_Advection_diffusion_set_properties,
     METH_VARARGS,
     pyRegional_Advection_diffusion_set_properties__doc__},

    {pyRegional_BC_set_properties__name__,
     pyRegional_BC_set_properties,
     METH_VARARGS,
     pyRegional_BC_set_properties__doc__},

    {pyRegional_Const_set_properties__name__,
     pyRegional_Const_set_properties,
     METH_VARARGS,
     pyRegional_Const_set_properties__doc__},

    {pyRegional_IC_set_properties__name__,
     pyRegional_IC_set_properties,
     METH_VARARGS,
     pyRegional_IC_set_properties__doc__},

    {pyRegional_Parallel_set_properties__name__,
     pyRegional_Parallel_set_properties,
     METH_VARARGS,
     pyRegional_Parallel_set_properties__doc__},

    {pyRegional_Param_set_properties__name__,
     pyRegional_Param_set_properties,
     METH_VARARGS,
     pyRegional_Param_set_properties__doc__},

    {pyRegional_Phase_set_properties__name__,
     pyRegional_Phase_set_properties,
     METH_VARARGS,
     pyRegional_Phase_set_properties__doc__},

    {pyRegional_RegionalSphere_set_properties__name__,
     pyRegional_RegionalSphere_set_properties,
     METH_VARARGS,
     pyRegional_RegionalSphere_set_properties__doc__},

    {pyRegional_Visc_set_properties__name__,
     pyRegional_Visc_set_properties,
     METH_VARARGS,
     pyRegional_Visc_set_properties__doc__},

    {pyRegional_Stokes_solver_set_properties__name__,
     pyRegional_Stokes_solver_set_properties,
     METH_VARARGS,
     pyRegional_Stokes_solver_set_properties__doc__},

    // from stokes_solver.h

    {pyRegional_assemble_forces__name__,
     pyRegional_assemble_forces,
     METH_VARARGS,
     pyRegional_assemble_forces__doc__},

    {pyRegional_construct_stiffness_B_matrix__name__,
     pyRegional_construct_stiffness_B_matrix,
     METH_VARARGS,
     pyRegional_construct_stiffness_B_matrix__doc__},

    {pyRegional_general_stokes_solver_Unorm__name__,
     pyRegional_general_stokes_solver_Unorm,
     METH_VARARGS,
     pyRegional_general_stokes_solver_Unorm__doc__},

//     {pyRegional_general_stokes_solver_fini__name__,
//      pyRegional_general_stokes_solver_fini,
//      METH_VARARGS,
//      pyRegional_general_stokes_solver_fini__doc__},

    {pyRegional_general_stokes_solver_init__name__,
     pyRegional_general_stokes_solver_init,
     METH_VARARGS,
     pyRegional_general_stokes_solver_init__doc__},

    {pyRegional_general_stokes_solver_log__name__,
     pyRegional_general_stokes_solver_log,
     METH_VARARGS,
     pyRegional_general_stokes_solver_log__doc__},

    {pyRegional_general_stokes_solver_update_velo__name__,
     pyRegional_general_stokes_solver_update_velo,
     METH_VARARGS,
     pyRegional_general_stokes_solver_update_velo__doc__},

    {pyRegional_get_system_viscosity__name__,
     pyRegional_get_system_viscosity,
     METH_VARARGS,
     pyRegional_get_system_viscosity__doc__},

    {pyRegional_set_cg_defaults__name__,
     pyRegional_set_cg_defaults,
     METH_VARARGS,
     pyRegional_set_cg_defaults__doc__},

    {pyRegional_set_mg_defaults__name__,
     pyRegional_set_mg_defaults,
     METH_VARARGS,
     pyRegional_set_mg_defaults__doc__},

    {pyRegional_set_mg_el_defaults__name__,
     pyRegional_set_mg_el_defaults,
     METH_VARARGS,
     pyRegional_set_mg_el_defaults__doc__},

    {pyRegional_solve_constrained_flow_iterative__name__,
     pyRegional_solve_constrained_flow_iterative,
     METH_VARARGS,
     pyRegional_solve_constrained_flow_iterative__doc__},


// Sentinel
    {0, 0, 0, 0}
};

// version
// $Id: bindings.cc,v 1.23 2003/07/28 21:57:01 tan2 Exp $

// End of file
