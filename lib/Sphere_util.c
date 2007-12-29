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

/* Common functions relating to the building and use of mesh locations ... */

#include <math.h>
#include "global_defs.h"

/* =================================================
  this routine evenly divides the arc between points
  1 and 2 in a great cicle. The word "evenly" means
  anglewise evenly.
 =================================================*/

void even_divide_arc12(elx,x1,y1,z1,x2,y2,z2,theta,fi)
 double x1,y1,z1,x2,y2,z2,*theta,*fi;
 int elx;
{
  double dx,dy,dz,xx,yy,zz,myatan();
  int j, nox;

  nox = elx+1;

  dx = (x2 - x1)/elx;
  dy = (y2 - y1)/elx;
  dz = (z2 - z1)/elx;
  for (j=1;j<=nox;j++)   {
      xx = x1 + dx*(j-1) + 5.0e-32;
      yy = y1 + dy*(j-1);
      zz = z1 + dz*(j-1);
      theta[j] = acos(zz/sqrt(xx*xx+yy*yy+zz*zz));
      fi[j]    = myatan(yy,xx);
      }

   return;
  }


/* =================================================
  rotate the mesh
 =================================================*/
void rotate_mesh(E,m,icap)
   struct All_variables *E;
   int icap,m;
  {

  int i,lev;
  double t[4],myatan();

  for (lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++)  {
    for (i=1;i<=E->lmesh.NNO[lev];i++)  {
      t[0] = E->X[lev][m][1][i]*E->sphere.dircos[1][1]+
             E->X[lev][m][2][i]*E->sphere.dircos[1][2]+
             E->X[lev][m][3][i]*E->sphere.dircos[1][3];
      t[1] = E->X[lev][m][1][i]*E->sphere.dircos[2][1]+
             E->X[lev][m][2][i]*E->sphere.dircos[2][2]+
             E->X[lev][m][3][i]*E->sphere.dircos[2][3];
      t[2] = E->X[lev][m][1][i]*E->sphere.dircos[3][1]+
             E->X[lev][m][2][i]*E->sphere.dircos[3][2]+
             E->X[lev][m][3][i]*E->sphere.dircos[3][3];

      E->X[lev][m][1][i] = t[0];
      E->X[lev][m][2][i] = t[1];
      E->X[lev][m][3][i] = t[2];
      E->SX[lev][m][1][i] = acos(t[2]/E->SX[lev][m][3][i]);
      E->SX[lev][m][2][i] = myatan(t[1],t[0]);
      }
    }    /* lev */

  return;
  }
