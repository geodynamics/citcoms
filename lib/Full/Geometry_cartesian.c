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
#include <math.h>
#include "element_definitions.h"
#include "global_defs.h"



void full_set_2dc_defaults(E)
     struct All_variables *E;
{ 

  E->mesh.nsd = 2;
  E->mesh.dof = 2;
  
}


void full_set_2pt5dc_defaults(E)  
    struct All_variables *E;
{ 

  E->mesh.nsd = 2;
  E->mesh.dof = 3;
 
}

void full_set_3dc_defaults(E)
     struct All_variables *E;
{ 

  E->mesh.nsd = 3;
  E->mesh.dof = 3;
 
}

void full_set_3dsphere_defaults(E)
     struct All_variables *E;
{ 
  int i,j;
  double offset;
  int m=E->parallel.me;

  E->mesh.nsd = 3;
  E->mesh.dof = 3;

  E->sphere.caps = 12;
  E->sphere.max_connections = 6;

  input_double("radius_outer",&(E->sphere.ro),"essential",m);
  input_double("radius_inner",&(E->sphere.ri),"essential",m);

  offset = 10.0/180.0*M_PI;

  for (i=1;i<=4;i++)  {
    E->sphere.cap[(i-1)*3+1].theta[1] = 0.0;
    E->sphere.cap[(i-1)*3+1].theta[2] = M_PI/4.0+offset;
    E->sphere.cap[(i-1)*3+1].theta[3] = M_PI/2.0;
    E->sphere.cap[(i-1)*3+1].theta[4] = M_PI/4.0+offset;
    E->sphere.cap[(i-1)*3+1].fi[1] = 0.0;
    E->sphere.cap[(i-1)*3+1].fi[2] = (i-1)*M_PI/2.0;
    E->sphere.cap[(i-1)*3+1].fi[3] = (i-1)*M_PI/2.0 + M_PI/4.0;
    E->sphere.cap[(i-1)*3+1].fi[4] = i*M_PI/2.0;

    E->sphere.cap[(i-1)*3+2].theta[1] = M_PI/4.0+offset;
    E->sphere.cap[(i-1)*3+2].theta[2] = M_PI/2.0;
    E->sphere.cap[(i-1)*3+2].theta[3] = 3*M_PI/4.0-offset;
    E->sphere.cap[(i-1)*3+2].theta[4] = M_PI/2.0;
    E->sphere.cap[(i-1)*3+2].fi[1] = i*M_PI/2.0;
    E->sphere.cap[(i-1)*3+2].fi[2] = i*M_PI/2.0 - M_PI/4.0;
    E->sphere.cap[(i-1)*3+2].fi[3] = i*M_PI/2.0;
    E->sphere.cap[(i-1)*3+2].fi[4] = i*M_PI/2.0 + M_PI/4.0;
    }

  for (i=1;i<=4;i++)  {
    j = (i-1)*3;
    if (i==1) j=12;
    E->sphere.cap[j].theta[1] = M_PI/2.0;
    E->sphere.cap[j].theta[2] = 3*M_PI/4.0-offset;
    E->sphere.cap[j].theta[3] = M_PI;
    E->sphere.cap[j].theta[4] = 3*M_PI/4.0-offset;
    E->sphere.cap[j].fi[1] = (i-1)*M_PI/2.0 + M_PI/4.0;
    E->sphere.cap[j].fi[2] = (i-1)*M_PI/2.0;
    E->sphere.cap[j].fi[3] = 0.0;
    E->sphere.cap[j].fi[4] = i*M_PI/2.0;
    }

  return;
 }
