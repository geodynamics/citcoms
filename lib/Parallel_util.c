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


#include <mpi.h>
#include <stdlib.h>
#include "global_defs.h"

/* ============================================ */
/* ============================================ */

PetscErrorCode parallel_process_finalize()
{
  return PetscFinalize();
}

/* ============================================ */
/* ============================================ */

void parallel_process_termination()
{

  PetscFinalize();
  exit(8);
}

/* ============================================ */
/* ============================================ */

void parallel_process_sync(struct All_variables *E)
{

  MPI_Barrier(E->parallel.world);
}


/* ==========================   */

double CPU_time0()
{
  double time, MPI_Wtime();
  time = MPI_Wtime();
  return (time);
}

/* End of file */
