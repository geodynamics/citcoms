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
#include "element_definitions.h"
#include "global_defs.h"
#include <math.h>

#include "lith_age.h"

/* ========================================== */

static void horizontal_bc();
static void velocity_apply_periodic_bcs();
static void temperature_apply_periodic_bcs();

/* ========================================== */

void full_velocity_boundary_conditions(E)
     struct All_variables *E;
{
  void velocity_imp_vert_bc();
  void velocity_apply_periodicapply_periodic_bcs();
  void read_velocity_boundary_from_file();
  void apply_side_sbc();

  int j,noz,lv;

  for(lv=E->mesh.gridmax;lv>=E->mesh.gridmin;lv--)
    for (j=1;j<=E->sphere.caps_per_proc;j++)     {
      noz = E->mesh.NOZ[lv];
      if(E->mesh.topvbc != 1) {
	horizontal_bc(E,E->sphere.cap[j].VB,noz,1,0.0,VBX,0,lv,j);
	horizontal_bc(E,E->sphere.cap[j].VB,noz,3,0.0,VBZ,1,lv,j);
	horizontal_bc(E,E->sphere.cap[j].VB,noz,2,0.0,VBY,0,lv,j);
	horizontal_bc(E,E->sphere.cap[j].VB,noz,1,E->control.VBXtopval,SBX,1,lv,j);
	horizontal_bc(E,E->sphere.cap[j].VB,noz,3,0.0,SBZ,0,lv,j);
	horizontal_bc(E,E->sphere.cap[j].VB,noz,2,E->control.VBYtopval,SBY,1,lv,j);
	}
      if(E->mesh.botvbc != 1) {
        horizontal_bc(E,E->sphere.cap[j].VB,1,1,0.0,VBX,0,lv,j);
        horizontal_bc(E,E->sphere.cap[j].VB,1,3,0.0,VBZ,1,lv,j);
        horizontal_bc(E,E->sphere.cap[j].VB,1,2,0.0,VBY,0,lv,j);
        horizontal_bc(E,E->sphere.cap[j].VB,1,1,E->control.VBXbotval,SBX,1,lv,j);
        horizontal_bc(E,E->sphere.cap[j].VB,1,3,0.0,SBZ,0,lv,j);
        horizontal_bc(E,E->sphere.cap[j].VB,1,2,E->control.VBYbotval,SBY,1,lv,j);
        }

      if(E->mesh.topvbc == 1) {
        horizontal_bc(E,E->sphere.cap[j].VB,noz,1,E->control.VBXtopval,VBX,1,lv,j);
        horizontal_bc(E,E->sphere.cap[j].VB,noz,3,0.0,VBZ,1,lv,j);
        horizontal_bc(E,E->sphere.cap[j].VB,noz,2,E->control.VBYtopval,VBY,1,lv,j);
        horizontal_bc(E,E->sphere.cap[j].VB,noz,1,0.0,SBX,0,lv,j);
        horizontal_bc(E,E->sphere.cap[j].VB,noz,3,0.0,SBZ,0,lv,j);
        horizontal_bc(E,E->sphere.cap[j].VB,noz,2,0.0,SBY,0,lv,j);

        if(E->control.vbcs_file)
          read_velocity_boundary_from_file(E);

        }
      if(E->mesh.botvbc == 1) {
        horizontal_bc(E,E->sphere.cap[j].VB,1,1,E->control.VBXbotval,VBX,1,lv,j);
        horizontal_bc(E,E->sphere.cap[j].VB,1,3,0.0,VBZ,1,lv,j);
        horizontal_bc(E,E->sphere.cap[j].VB,1,2,E->control.VBYbotval,VBY,1,lv,j);
        horizontal_bc(E,E->sphere.cap[j].VB,1,1,0.0,SBX,0,lv,j);
        horizontal_bc(E,E->sphere.cap[j].VB,1,3,0.0,SBZ,0,lv,j);
        horizontal_bc(E,E->sphere.cap[j].VB,1,2,0.0,SBY,0,lv,j);
        }
      }    /* end for j and lv */

      if(E->control.side_sbcs)
	apply_side_sbc(E);

/* if(E->control.verbose) */
/*  for (j=1;j<=E->sphere.caps_per_proc;j++) */
/*    for (node=1;node<=E->lmesh.nno;node++) */
/*       fprintf(E->fp_out,"m=%d VB== %d %g %g %g flag %u %u %u\n",j,node,E->sphere.cap[j].VB[1][node],E->sphere.cap[j].VB[2][node],E->sphere.cap[j].VB[3][node],E->node[j][node]&VBX,E->node[j][node]&VBY,E->node[j][node]&VBZ); */

  /* If any imposed internal velocity structure it goes here */


   return; }

/* ========================================== */

void full_temperature_boundary_conditions(E)
     struct All_variables *E;
{
  void temperatures_conform_bcs();
  void temperature_imposed_vert_bcs();
  int j,lev,noz;

  lev = E->mesh.levmax;
  for (j=1;j<=E->sphere.caps_per_proc;j++)    {
    noz = E->mesh.noz;
    if(E->mesh.toptbc == 1)    {
      horizontal_bc(E,E->sphere.cap[j].TB,noz,3,E->control.TBCtopval,TBZ,1,lev,j);
      horizontal_bc(E,E->sphere.cap[j].TB,noz,3,E->control.TBCtopval,FBZ,0,lev,j);
      }
    else   {
      horizontal_bc(E,E->sphere.cap[j].TB,noz,3,E->control.TBCtopval,TBZ,0,lev,j);
      horizontal_bc(E,E->sphere.cap[j].TB,noz,3,E->control.TBCtopval,FBZ,1,lev,j);
      }

    if(E->mesh.bottbc == 1)    {
      horizontal_bc(E,E->sphere.cap[j].TB,1,3,E->control.TBCbotval,TBZ,1,lev,j);
      horizontal_bc(E,E->sphere.cap[j].TB,1,3,E->control.TBCbotval,FBZ,0,lev,j);
      }
    else        {
      horizontal_bc(E,E->sphere.cap[j].TB,1,3,E->control.TBCbotval,TBZ,0,lev,j);
      horizontal_bc(E,E->sphere.cap[j].TB,1,3,E->control.TBCbotval,FBZ,1,lev,j);
      }

    }     /* end for j */

  temperatures_conform_bcs(E);
  E->temperatures_conform_bcs = temperatures_conform_bcs;

   return; }


/*  =========================================================  */


static void horizontal_bc(E,BC,ROW,dirn,value,mask,onoff,level,m)
     struct All_variables *E;
     float *BC[];
     int ROW;
     int dirn;
     float value;
     unsigned int mask;
     char onoff;
     int level,m;

{
  int i,j,node,rowl;

    /* safety feature */
  if(dirn > E->mesh.nsd)
     return;

  if (ROW==1)
      rowl = 1;
  else
      rowl = E->lmesh.NOZ[level];

  if ( ( (ROW==1) && (E->parallel.me_loc[3]==0) ) ||
       ( (ROW==E->mesh.NOZ[level]) && (E->parallel.me_loc[3]==E->parallel.nprocz-1) ) ) {

    /* turn bc marker to zero */
    if (onoff == 0)          {
      for(j=1;j<=E->lmesh.NOY[level];j++)
    	for(i=1;i<=E->lmesh.NOX[level];i++)     {
    	  node = rowl+(i-1)*E->lmesh.NOZ[level]+(j-1)*E->lmesh.NOX[level]*E->lmesh.NOZ[level];
    	  E->NODE[level][m][node] = E->NODE[level][m][node] & (~ mask);
    	  }        /* end for loop i & j */
      }

    /* turn bc marker to one */
    else        {
      for(j=1;j<=E->lmesh.NOY[level];j++)
        for(i=1;i<=E->lmesh.NOX[level];i++)       {
    	  node = rowl+(i-1)*E->lmesh.NOZ[level]+(j-1)*E->lmesh.NOX[level]*E->lmesh.NOZ[level];
    	  E->NODE[level][m][node] = E->NODE[level][m][node] | (mask);
    	  if(level==E->mesh.levmax)   /* NB */
    	    BC[dirn][node] = value;
    	  }     /* end for loop i & j */
      }

    }             /* end for if ROW */

  return;
}


static void velocity_apply_periodic_bcs(E)
    struct All_variables *E;
{
  fprintf(E->fp,"Periodic boundary conditions\n");

  return;
  }

static void temperature_apply_periodic_bcs(E)
    struct All_variables *E;
{
 fprintf(E->fp,"Periodic temperature boundary conditions\n");

  return;
  }



/* version */
/* $Id$ */

/* End of file  */
