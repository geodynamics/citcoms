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

/* 

assign boundary conditions to a horizontal layer of nodes

row > 0: regular operation, only assign BCs to bottom (row = 1) and
         top (row = E->lmesh.NOZ[level]) processors, as in the
         previous version where horizontal_bc was a function in both
         Full and Regional BC subroutines

row < 0: assign BCs to -row, no matter what processor, to allow
         setting internal boundary conditions

 */
void horizontal_bc(E,BC,row,dirn,value,mask,onoff,level,m)
     struct All_variables *E;
     float *BC[];
     int row;
     int dirn;
     float value;
     unsigned int mask;
     char onoff;
     int level,m;

{
  int i,j,node,noxnoz;
  static short int warned = FALSE;
    /* safety feature */
  if(dirn > E->mesh.nsd)
     return;
  
  noxnoz = E->lmesh.NOX[level]*E->lmesh.NOZ[level];
 
  if(row < 0){
    /* 
       assignment to any row
    */
    row = -row;
    if((row !=  E->lmesh.NOZ[level]) && (row != 1) && (!warned)){
      fprintf(stderr,"horizontal_bc: CPU %4i: assigning internal BC, row: %4i nozl: %4i noz: %4i\n",
	      E->parallel.me,row, E->lmesh.NOZ[level], E->mesh.noz);
      warned = TRUE;
    }
    if((row >  E->lmesh.NOZ[level])||(row<1))
      myerror(E,"horizontal_bc: error, out of bounds");

    /* turn bc marker to zero */
    if (onoff == 0)          {
      for(j=1;j<=E->lmesh.NOY[level];j++)
	for(i=1;i<=E->lmesh.NOX[level];i++)     {
	  node = row+(i-1)*E->lmesh.NOZ[level]+(j-1)*noxnoz;
	  E->NODE[level][m][node] = E->NODE[level][m][node] & (~ mask);
	}        /* end for loop i & j */
    }
    
    /* turn bc marker to one */
    else        {
      for(j=1;j<=E->lmesh.NOY[level];j++)
	for(i=1;i<=E->lmesh.NOX[level];i++)       {
	  node = row+(i-1)*E->lmesh.NOZ[level]+(j-1)*noxnoz;
	  E->NODE[level][m][node] = E->NODE[level][m][node] | (mask);
	  if(level==E->mesh.levmax)   /* NB */
	    BC[dirn][node] = value;
	}     /* end for loop i & j */
    }
  }else{
    /* regular operation, assign only if in top
       (row==E->lmesh.NOZ[level]) or bottom (row==1) processor  */
    
    if ( ( (row==1) && (E->parallel.me_loc[3]==0) ) || /* bottom or top */
	 ( (row==E->lmesh.NOZ[level]) && (E->parallel.me_loc[3]==E->parallel.nprocz-1) ) ) {
      
      /* turn bc marker to zero */
      if (onoff == 0)          {
	for(j=1;j<=E->lmesh.NOY[level];j++)
	  for(i=1;i<=E->lmesh.NOX[level];i++)     {
	    node = row+(i-1)*E->lmesh.NOZ[level]+(j-1)*noxnoz;
	    E->NODE[level][m][node] = E->NODE[level][m][node] & (~ mask);
    	  }        /* end for loop i & j */
      }
      
      /* turn bc marker to one */
      else        {
	for(j=1;j<=E->lmesh.NOY[level];j++)
	  for(i=1;i<=E->lmesh.NOX[level];i++)       {
	    node = row+(i-1)*E->lmesh.NOZ[level]+(j-1)*noxnoz;
	    E->NODE[level][m][node] = E->NODE[level][m][node] | (mask);
	    if(level==E->mesh.levmax)   /* NB */
	      BC[dirn][node] = value;
    	  }     /* end for loop i & j */
      }
      
    }             /* end for if row */
  }
  return;
}



void strip_bcs_from_residual(E,Res,level)
    struct All_variables *E;
    double **Res;
    int level;
{
    int m,i;

  for (m=1;m<=E->sphere.caps_per_proc;m++)
    if (E->num_zero_resid[level][m])
      for(i=1;i<=E->num_zero_resid[level][m];i++)
         Res[m][E->zero_resid[level][m][i]] = 0.0;

    return;
}


void temperatures_conform_bcs(E)
     struct All_variables *E;
{
  void temperatures_conform_bcs2(struct All_variables *);
  void assimilate_lith_conform_bcs2(struct All_variables *);

  if(E->control.lith_age) {
    /*
    This sequence now moved to end of PG_time_step_solve
    lith_age_conform_tbc(E);
    assimilate_lith_conform_bcs(E);
    */
    }
  else
    temperatures_conform_bcs2(E);
  return;
}


void temperatures_conform_bcs2(E)
     struct All_variables *E;
{
  int j,node;
  unsigned int type;

  for(j=1;j<=E->sphere.caps_per_proc;j++)
    for(node=1;node<=E->lmesh.nno;node++)  {

        type = (E->node[j][node] & (TBX | TBZ | TBY));

        switch (type) {
        case 0:  /* no match, next node */
            break;
        case TBX:
            E->T[j][node] = E->sphere.cap[j].TB[1][node];
            break;
        case TBZ:
            E->T[j][node] = E->sphere.cap[j].TB[3][node];
            break;
        case TBY:
            E->T[j][node] = E->sphere.cap[j].TB[2][node];
            break;
        case (TBX | TBZ):     /* clashes ! */
            E->T[j][node] = 0.5 * (E->sphere.cap[j].TB[1][node] + E->sphere.cap[j].TB[3][node]);
            break;
        case (TBX | TBY):     /* clashes ! */
            E->T[j][node] = 0.5 * (E->sphere.cap[j].TB[1][node] + E->sphere.cap[j].TB[2][node]);
            break;
        case (TBZ | TBY):     /* clashes ! */
            E->T[j][node] = 0.5 * (E->sphere.cap[j].TB[3][node] + E->sphere.cap[j].TB[2][node]);
            break;
        case (TBZ | TBY | TBX):     /* clashes ! */
            E->T[j][node] = 0.3333333 * (E->sphere.cap[j].TB[1][node] + E->sphere.cap[j].TB[2][node] + E->sphere.cap[j].TB[3][node]);
            break;
        }

        /* next node */
    }

  return;

}


void velocities_conform_bcs(E,U)
    struct All_variables *E;
    double **U;
{
    int node,m;

    const unsigned int typex = VBX;
    const unsigned int typez = VBZ;
    const unsigned int typey = VBY;

    const int nno = E->lmesh.nno;

    for(m=1;m<=E->sphere.caps_per_proc;m++)   {
      for(node=1;node<=nno;node++) {

        if (E->node[m][node] & typex)
	      U[m][E->id[m][node].doff[1]] = E->sphere.cap[m].VB[1][node];
 	if (E->node[m][node] & typey)
	      U[m][E->id[m][node].doff[2]] = E->sphere.cap[m].VB[2][node];
	if (E->node[m][node] & typez)
	      U[m][E->id[m][node].doff[3]] = E->sphere.cap[m].VB[3][node];
        }
      }

    return;
}

/* 

facility to apply internal velocity or stress conditions after
top/bottom

options:

toplayerbc  > 0: assign surface BC down to toplayerbc nd
toplayerbc == 0: no action

 */
void assign_internal_bc(struct All_variables *E,int is_global)
{
  
  int lv, j, noz, k,node,lay;
  /* stress or vel BC within a layer */
  

  if(E->mesh.toplayerbc > 0){
    for(lv=E->mesh.gridmax;lv>=E->mesh.gridmin;lv--)
      for (j=1;j<=E->sphere.caps_per_proc;j++)     {
	noz = E->lmesh.NOZ[lv];
	for(k=noz-1;k >= 1;k--){ /* assumes regular grid */
	  node = k;		/* global node number */
	  if((lay = layers(E,j,node)) <= E->mesh.toplayerbc){
	    if(E->mesh.topvbc != 1) {	/* free slip top */
	      horizontal_bc(E,E->sphere.cap[j].VB,-k,1,0.0,VBX,0,lv,j);
	      horizontal_bc(E,E->sphere.cap[j].VB,-k,3,0.0,VBZ,1,lv,j);
	      horizontal_bc(E,E->sphere.cap[j].VB,-k,2,0.0,VBY,0,lv,j);
	      horizontal_bc(E,E->sphere.cap[j].VB,-k,1,E->control.VBXtopval,SBX,1,lv,j);
	      horizontal_bc(E,E->sphere.cap[j].VB,-k,3,0.0,SBZ,0,lv,j);
	      horizontal_bc(E,E->sphere.cap[j].VB,-k,2,E->control.VBYtopval,SBY,1,lv,j);
	    }else{		/* no slip */
	      horizontal_bc(E,E->sphere.cap[j].VB,-k,1,E->control.VBXtopval,VBX,1,lv,j);
	      horizontal_bc(E,E->sphere.cap[j].VB,-k,3,0.0,VBZ,1,lv,j);
	      horizontal_bc(E,E->sphere.cap[j].VB,-k,2,E->control.VBYtopval,VBY,1,lv,j);
	      horizontal_bc(E,E->sphere.cap[j].VB,-k,1,0.0,SBX,0,lv,j);
	      horizontal_bc(E,E->sphere.cap[j].VB,-k,3,0.0,SBZ,0,lv,j);
	      horizontal_bc(E,E->sphere.cap[j].VB,-k,2,0.0,SBY,0,lv,j);
	    }
	  }
	}
      }
    /* read in velocities/stresses from grd file? */
#ifdef USE_GGRD
    if(E->control.ggrd.vtop_control)
      ggrd_read_vtop_from_file(E, is_global, 1);
#endif
  } /* end toplayerbc > 0 branch */
}



/* End of file  */

