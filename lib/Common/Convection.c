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
    int m=E->parallel.me;

    /* parameters */

    input_float("rayleigh",&(E->control.Atemp),"essential",m);
    input_float("inputdiffusivity",&(E->control.inputdiff),"1.0",m);

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
    void temperatures_conform_bcs();

    velocity_boundary_conditions(E);      /* universal */
    temperature_boundary_conditions(E);

    return;
}

