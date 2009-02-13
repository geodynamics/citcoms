/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 *<LicenseText>
 *
 * CitcomS by Louis Moresi, Shijie Zhong, Lijie Han, Eh Tan,
 * Clint Conrad, Michael Gurnis, and Eun-seo Choi.
 * Copyright (C) 1994-2005, California Institute of Technology.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *</LicenseText>
 *
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
#include "global_defs.h"
#include "parsing.h"

#include "cproto.h"

void regional_set_2dc_defaults(struct All_variables *E)
{

  E->mesh.nsd = 2;
  E->mesh.dof = 2;

}


void regional_set_2pt5dc_defaults(struct All_variables *E)
{

  E->mesh.nsd = 2;
  E->mesh.dof = 3;

}

void regional_set_3dc_defaults(struct All_variables *E)
{

  E->mesh.nsd = 3;
  E->mesh.dof = 3;

}

void regional_set_3dsphere_defaults(struct All_variables *E)
{
  int m = E->parallel.me;

  input_double("radius_outer",&(E->sphere.ro),"1",m);
  input_double("radius_inner",&(E->sphere.ri),"0.55",m);

  input_double("theta_min",&(E->control.theta_min),"essential",m);
  input_double("theta_max",&(E->control.theta_max),"essential",m);
  input_double("fi_min",&(E->control.fi_min),"essential",m);
  input_double("fi_max",&(E->control.fi_max),"essential",m);

  regional_set_3dsphere_defaults2(E);

  return;
}


void regional_set_3dsphere_defaults2(struct All_variables *E)
{
  E->mesh.nsd = 3;
  E->mesh.dof = 3;

  E->sphere.caps = 1;
  E->sphere.max_connections = 6;

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
