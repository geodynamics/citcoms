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
