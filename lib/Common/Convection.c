/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * 
 *<LicenseText>
 *=====================================================================
 *
 *                              CitcomS
 *                 ---------------------------------
 *
 *                              Authors:
 *           Louis Moresi, Shijie Zhong, Lijie Han, Eh Tan,
 *           Clint Conrad, Michael Gurnis, and Eun-seo Choi
 *          (c) California Institute of Technology 1994-2005
 *
 *        By downloading and/or installing this software you have
 *       agreed to the CitcomS.py-LICENSE bundled with this software.
 *             Free for non-commercial academic research ONLY.
 *      This program is distributed WITHOUT ANY WARRANTY whatsoever.
 *
 *=====================================================================
 *
 *  Copyright June 2005, by the California Institute of Technology.
 *  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
 * 
 *  Any commercial use must be negotiated with the Office of Technology
 *  Transfer at the California Institute of Technology. This software
 *  may be subject to U.S. export control laws and regulations. By
 *  accepting this software, the user agrees to comply with all
 *  applicable U.S. export laws and regulations, including the
 *  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
 *  the Export Administration Regulations, 15 C.F.R. 730-744. User has
 *  the responsibility to obtain export licenses, or other export
 *  authority as may be required before exporting such information to
 *  foreign countries or providing access to foreign nationals.  In no
 *  event shall the California Institute of Technology be liable to any
 *  party for direct, indirect, special, incidental or consequential
 *  damages, including lost profits, arising out of the use of this
 *  software and its documentation, even if the California Institute of
 *  Technology has been advised of the possibility of such damage.
 * 
 *  The California Institute of Technology specifically disclaims any
 *  warranties, including the implied warranties or merchantability and
 *  fitness for a particular purpose. The software and documentation
 *  provided hereunder is on an "as is" basis, and the California
 *  Institute of Technology has no obligations to provide maintenance,
 *  support, updates, enhancements or modifications.
 *
 *=====================================================================
 *</LicenseText>
 * 
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
/* Assumes parameter list is opened and reads the things it needs.
   Variables are initialized etc, default values are set */


#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"
#include <stdlib.h> /* for "system" command */
#include <strings.h>

void set_convection_defaults(E)
     struct All_variables *E;
{
    void PG_timestep_with_melting();
    void PG_timestep();
    void read_convection_settings();
    void convection_derived_values();
    void convection_allocate_memory();
    void convection_boundary_conditions();
    void node_locations();
    void convection_initial_fields();
    void twiddle_thumbs();

    E->next_buoyancy_field = PG_timestep;
    E->special_process_new_buoyancy = twiddle_thumbs;
    E->problem_settings = read_convection_settings;
    E->problem_derived_values = convection_derived_values;
    E->problem_allocate_vars = convection_allocate_memory;
    E->problem_boundary_conds = convection_boundary_conditions;
    E->problem_initial_fields = convection_initial_fields;
    E->problem_node_positions = node_locations;
    E->problem_update_node_positions = twiddle_thumbs;
    E->problem_update_bcs = twiddle_thumbs;

/*     sprintf(E->control.which_data_files,"Temp,Strf,Pres"); */
/*     sprintf(E->control.which_horiz_averages,"Temp,Visc,Vrms"); */
/*     sprintf(E->control.which_running_data,"Step,Time,"); */
/*     sprintf(E->control.which_observable_data,"Shfl"); */

    return;
}

void read_convection_settings(E)
     struct All_variables *E;

{
    void advection_diffusion_parameters();

    /* parameters */

    advection_diffusion_parameters(E);

    return;
}

/* =================================================================
   Any setup which relates only to the convection stuff goes in here
   ================================================================= */

void convection_derived_values(E)
     struct All_variables *E;

{

  return;
}

void convection_allocate_memory(E)
     struct All_variables *E;

{ void advection_diffusion_allocate_memory();

  advection_diffusion_allocate_memory(E);

  return;
}

/* ============================================ */

void convection_initial_fields(E)
     struct All_variables *E;

{
    void convection_initial_temperature();

    convection_initial_temperature(E);

  return; }

/* =========================================== */

void convection_boundary_conditions(E)
     struct All_variables *E;

{
    void velocity_boundary_conditions();
    void temperature_boundary_conditions();

    velocity_boundary_conditions(E);      /* universal */
    temperature_boundary_conditions(E);

    return;
}

