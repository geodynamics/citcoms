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

struct PyMethodDef pyCitcom_methods[] = {

    // dummy entry for testing
    {pyCitcom_copyright__name__,
     pyCitcom_copyright,
     METH_VARARGS,
     pyCitcom_copyright__doc__},


    //////////////////////////////////////////////////////////////////////////
    // This section is for testing or temporatory implementation
    //////////////////////////////////////////////////////////////////////////

    {pyCitcom_return1_test__name__,
     pyCitcom_return1_test,
     METH_VARARGS,
     pyCitcom_return1_test__doc__},

    {pyCitcom_CPU_time__name__,
     pyCitcom_CPU_time,
     METH_VARARGS,
     pyCitcom_CPU_time__doc__},

    {pyCitcom_read_instructions__name__,
     pyCitcom_read_instructions,
     METH_VARARGS,
     pyCitcom_read_instructions__doc__},


    //////////////////////////////////////////////////////////////////////////
    // This section is for finished implementation
    //////////////////////////////////////////////////////////////////////////

    // from misc.h

    {pyCitcom_citcom_init__name__,
     pyCitcom_citcom_init,
     METH_VARARGS,
     pyCitcom_citcom_init__doc__},

    {pyCitcom_global_default_values__name__,
     pyCitcom_global_default_values,
     METH_VARARGS,
     pyCitcom_global_default_values__doc__},

    {pyCitcom_set_signal__name__,
     pyCitcom_set_signal,
     METH_VARARGS,
     pyCitcom_set_signal__doc__},

    {pyCitcom_velocities_conform_bcs__name__,
     pyCitcom_velocities_conform_bcs,
     METH_VARARGS,
     pyCitcom_velocities_conform_bcs__doc__},

    // from advdiffu.h

    {pyCitcom_PG_timestep_init__name__,
     pyCitcom_PG_timestep_init,
     METH_VARARGS,
     pyCitcom_PG_timestep_init__doc__},

    {pyCitcom_PG_timestep_solve__name__,
     pyCitcom_PG_timestep_solve,
     METH_VARARGS,
     pyCitcom_PG_timestep_solve__doc__},

    {pyCitcom_set_convection_defaults__name__,
     pyCitcom_set_convection_defaults,
     METH_VARARGS,
     pyCitcom_set_convection_defaults__doc__},

    {pyCitcom_stable_timestep__name__,
     pyCitcom_stable_timestep,
     METH_VARARGS,
     pyCitcom_stable_timestep__doc__},

    // from mesher.h

    {pyCitcom_full_sphere_launch__name__,
     pyCitcom_full_sphere_launch,
     METH_VARARGS,
     pyCitcom_full_sphere_launch__doc__},

    {pyCitcom_regional_sphere_launch__name__,
     pyCitcom_regional_sphere_launch,
     METH_VARARGS,
     pyCitcom_regional_sphere_launch__doc__},

    // from outputs.h

    {pyCitcom_output__name__,
     pyCitcom_output,
     METH_VARARGS,
     pyCitcom_output__doc__},

    // from setProperties.h

    {pyCitcom_Advection_diffusion_set_properties__name__,
     pyCitcom_Advection_diffusion_set_properties,
     METH_VARARGS,
     pyCitcom_Advection_diffusion_set_properties__doc__},

    {pyCitcom_BC_set_properties__name__,
     pyCitcom_BC_set_properties,
     METH_VARARGS,
     pyCitcom_BC_set_properties__doc__},

    {pyCitcom_Const_set_properties__name__,
     pyCitcom_Const_set_properties,
     METH_VARARGS,
     pyCitcom_Const_set_properties__doc__},

    {pyCitcom_IC_set_properties__name__,
     pyCitcom_IC_set_properties,
     METH_VARARGS,
     pyCitcom_IC_set_properties__doc__},

    {pyCitcom_Param_set_properties__name__,
     pyCitcom_Param_set_properties,
     METH_VARARGS,
     pyCitcom_Param_set_properties__doc__},

    {pyCitcom_Phase_set_properties__name__,
     pyCitcom_Phase_set_properties,
     METH_VARARGS,
     pyCitcom_Phase_set_properties__doc__},

    {pyCitcom_Solver_set_properties__name__,
     pyCitcom_Solver_set_properties,
     METH_VARARGS,
     pyCitcom_Solver_set_properties__doc__},

    {pyCitcom_Sphere_set_properties__name__,
     pyCitcom_Sphere_set_properties,
     METH_VARARGS,
     pyCitcom_Sphere_set_properties__doc__},

    {pyCitcom_Visc_set_properties__name__,
     pyCitcom_Visc_set_properties,
     METH_VARARGS,
     pyCitcom_Visc_set_properties__doc__},

    {pyCitcom_Incompressible_set_properties__name__,
     pyCitcom_Incompressible_set_properties,
     METH_VARARGS,
     pyCitcom_Incompressible_set_properties__doc__},

    // from stokes_solver.h

    {pyCitcom_assemble_forces__name__,
     pyCitcom_assemble_forces,
     METH_VARARGS,
     pyCitcom_assemble_forces__doc__},

    {pyCitcom_construct_stiffness_B_matrix__name__,
     pyCitcom_construct_stiffness_B_matrix,
     METH_VARARGS,
     pyCitcom_construct_stiffness_B_matrix__doc__},

    {pyCitcom_general_stokes_solver__name__,
     pyCitcom_general_stokes_solver,
     METH_VARARGS,
     pyCitcom_general_stokes_solver__doc__},

    {pyCitcom_get_system_viscosity__name__,
     pyCitcom_get_system_viscosity,
     METH_VARARGS,
     pyCitcom_get_system_viscosity__doc__},

    {pyCitcom_set_cg_defaults__name__,
     pyCitcom_set_cg_defaults,
     METH_VARARGS,
     pyCitcom_set_cg_defaults__doc__},

    {pyCitcom_set_mg_defaults__name__,
     pyCitcom_set_mg_defaults,
     METH_VARARGS,
     pyCitcom_set_mg_defaults__doc__},

    {pyCitcom_set_mg_el_defaults__name__,
     pyCitcom_set_mg_el_defaults,
     METH_VARARGS,
     pyCitcom_set_mg_el_defaults__doc__},

    {pyCitcom_solve_constrained_flow_iterative__name__,
     pyCitcom_solve_constrained_flow_iterative,
     METH_VARARGS,
     pyCitcom_solve_constrained_flow_iterative__doc__},


// Sentinel
    {0, 0, 0, 0}
};

// version
// $Id: bindings.cc,v 1.30 2003/10/28 23:52:44 tan2 Exp $

// End of file
