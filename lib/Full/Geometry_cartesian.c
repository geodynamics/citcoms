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
#include <math.h>
#include "element_definitions.h"
#include "global_defs.h"



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
