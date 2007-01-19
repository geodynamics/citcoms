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
/*

  Tracer_setup.c

      A program which initiates the distribution of tracers
      and advects those tracers in a time evolving velocity field.
      Called and used from the CitCOM finite element code.
      Written 2/96 M. Gurnis for Citcom in cartesian geometry
      Modified by Lijie in 1998 and by Vlad and Eh in 2005 for CitcomS

*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "global_defs.h"
#include "parsing.h"


void tracer_input(struct All_variables *E)
{
  int m=E->parallel.me;

  input_int("tracer",&(E->control.tracer),"0",m);
  input_string("tracer_file",E->control.tracer_file,"tracer.dat",m);
}


void tracer_initial_settings(E)
     struct All_variables *E;
{
   void regional_tracer_setup();
   void regional_tracer_advection();

   if(E->parallel.nprocxy == 1) {
       E->problem_tracer_setup = regional_tracer_setup;
       E->problem_tracer_advection = regional_tracer_advection;
   }
}
