#include "global_defs.h"
#include "parsing.h"


void set_2dc_defaults(E)
     struct All_variables *E;
{

  E->mesh.nsd = 2;
  E->mesh.dof = 2;

}


void set_2pt5dc_defaults(E)
    struct All_variables *E;
{

  E->mesh.nsd = 2;
  E->mesh.dof = 3;

}

void set_3dc_defaults(E)
     struct All_variables *E;
{

  E->mesh.nsd = 3;
  E->mesh.dof = 3;

}

void set_3dsphere_defaults(E)
     struct All_variables *E;
{
  int m = E->parallel.me;

  E->mesh.nsd = 3;
  E->mesh.dof = 3;

  E->sphere.caps = 1;
  E->sphere.max_connections = 6;

  input_double("radius_outer",&(E->sphere.ro),"essential",m);
  input_double("radius_inner",&(E->sphere.ri),"essential",m);

  input_double("theta_min",&(E->control.theta_min),"essential",m);
  input_double("theta_max",&(E->control.theta_max),"essential",m);
  input_double("fi_min",&(E->control.fi_min),"essential",m);
  input_double("fi_max",&(E->control.fi_max),"essential",m);

  E->sphere.cap[1].theta[1] = E->control.theta_min;
  E->sphere.cap[1].theta[2] = E->control.theta_max;
  E->sphere.cap[1].theta[3] = E->control.theta_max;
  E->sphere.cap[1].theta[4] = E->control.theta_min;
  E->sphere.cap[1].fi[1] = E->control.fi_min;
  E->sphere.cap[1].fi[2] = E->control.fi_min;
  E->sphere.cap[1].fi[3] = E->control.fi_max;
  E->sphere.cap[1].fi[4] = E->control.fi_max;

  return;
}
