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
#include "citcom_init.h"

struct All_variables* citcom_init(MPI_Comm *world)
{
  struct All_variables *E;
  int rank, nproc;

  E = (struct All_variables*) malloc(sizeof(struct All_variables));

  MPI_Comm_rank(*world, &rank);
  MPI_Comm_size(*world, &nproc);

  E->parallel.world = *world;
  E->parallel.nproc = nproc;
  E->parallel.me = rank;

  //fprintf(stderr,"%d in %d processpors\n", rank, nproc);

  E->monitor.solution_cycles=0;
  E->control.keep_going=1;

  E->control.total_iteration_cycles=0;
  E->control.total_v_solver_calls=0;

  return(E);
}
