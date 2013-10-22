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
void horizontal_bc(struct All_variables *,float *[],int,int,float,unsigned int,char,int,int);
void internal_horizontal_bc(struct All_variables *,float *[],int,int,float,unsigned int,char,int);
void myerror(struct All_variables *,char *);
int layers(struct All_variables *,int);


#ifdef USE_GGRD
#include "ggrd_handling.h"
#endif




/* 

assign boundary conditions to a horizontal layer of nodes within mesh,
without consideration of being in top or bottom processor

*/
void internal_horizontal_bc(struct All_variables *E,float *BC[],int row,
                            int dirn, float value,unsigned int mask,char onoff,
                            int level)
{
  int i,j,node,noxnoz;
  /* safety feature */
  if(dirn > E->mesh.nsd)
     return;
  
  noxnoz = E->lmesh.NOX[level]*E->lmesh.NOZ[level];
 
  /* 
     assignment to any row, any processor
  */
 
  if((row >  E->lmesh.NOZ[level])||(row < 1))
    myerror(E,"internal_horizontal_bc: error, row out of bounds");
  
  /* turn bc marker to zero */
  if (onoff == 0) {
    for(j=1;j<=E->lmesh.NOY[level];j++)
      for(i=1;i<=E->lmesh.NOX[level];i++) {
        node = row+(i-1)*E->lmesh.NOZ[level]+(j-1)*noxnoz;
        E->NODE[level][node] = E->NODE[level][node] & (~ mask);
      }        /* end for loop i & j */
  } else {
    /* turn bc marker to one */
    for(j=1;j<=E->lmesh.NOY[level];j++)
      for(i=1;i<=E->lmesh.NOX[level];i++) {
        node = row+(i-1)*E->lmesh.NOZ[level]+(j-1)*noxnoz;
        E->NODE[level][node] = E->NODE[level][node] | (mask);
        if(level == E->mesh.levmax)   /* NB */
          BC[dirn][node] = value;
      }     /* end for loop i & j */
  }
}

void strip_bcs_from_residual(E,Res,level)
    struct All_variables *E;
    double *Res;
    int level;
{
  int i;

  if( E->num_zero_resid[level] )
    for( i=1; i<=E->num_zero_resid[level]; i++ )
       Res[E->zero_resid[level][i]] = 0.0;
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

    for(node=1;node<=E->lmesh.nno;node++)  {

        type = (E->node[node] & (TBX | TBZ | TBY));

        switch (type) {
        case 0:  /* no match, next node */
            break;
        case TBX:
            E->T[node] = E->sphere.cap[1].TB[1][node];
            break;
        case TBZ:
            E->T[node] = E->sphere.cap[1].TB[3][node];
            break;
        case TBY:
            E->T[node] = E->sphere.cap[1].TB[2][node];
            break;
        case (TBX | TBZ):     /* clashes ! */
            E->T[node] = 0.5 * (E->sphere.cap[1].TB[1][node] + E->sphere.cap[1].TB[3][node]);
            break;
        case (TBX | TBY):     /* clashes ! */
            E->T[node] = 0.5 * (E->sphere.cap[1].TB[1][node] + E->sphere.cap[1].TB[2][node]);
            break;
        case (TBZ | TBY):     /* clashes ! */
            E->T[node] = 0.5 * (E->sphere.cap[1].TB[3][node] + E->sphere.cap[1].TB[2][node]);
            break;
        case (TBZ | TBY | TBX):     /* clashes ! */
            E->T[node] = 0.3333333 * (E->sphere.cap[1].TB[1][node] + E->sphere.cap[1].TB[2][node] + E->sphere.cap[1].TB[3][node]);
            break;
        }
        /* next node */
    }
}


void velocities_conform_bcs(E,U)
    struct All_variables *E;
    double *U;
{
    int node;

    const unsigned int typex = VBX;
    const unsigned int typez = VBZ;
    const unsigned int typey = VBY;

    const int nno = E->lmesh.nno;

    for(node=1;node<=nno;node++) {
      if (E->node[node] & typex)
        U[E->id[node].doff[1]] = E->sphere.cap[1].VB[1][node];
      if (E->node[node] & typey)
        U[E->id[node].doff[2]] = E->sphere.cap[1].VB[2][node];
      if (E->node[node] & typez)
        U[E->id[node].doff[3]] = E->sphere.cap[1].VB[3][node];
    }
}

/* 

facility to apply internal velocity or stress conditions after
top/bottom

options:

toplayerbc  > 0: assign surface boundary condition down to all nodes with r >= toplayerbc_r
toplayerbc == 0: no action
toplayerbc  < 0: assign surface boundary condition within medium at node -toplayerbc depth, ie.
                 toplayerbc = -1 is one node underneath surface

*/
void assign_internal_bc(struct All_variables *E)
{
  
  int lv, j, noz, k,lay,ncount,ontop,onbottom;
  /* stress or vel BC within a layer */
  ncount = 0;

  if(E->mesh.toplayerbc > 0){
    for(lv=E->mesh.gridmax;lv>=E->mesh.gridmin;lv--) {
      noz = E->lmesh.NOZ[lv];
      /* we're looping through all nodes for the possibility that
         there are several internal processors which need BCs */
      for(k=noz;k >= 1;k--){ /* assumes regular grid */
        ontop = ((k==noz) && (E->parallel.me_loc[3]==E->parallel.nprocz-1))?(1):(0);
        onbottom = ((k==1) && (E->parallel.me_loc[3]==0))?(1):(0);
        /* node number is k, assuming no dependence on x and y  */
        if(E->SX[lv][3][k] >= E->mesh.toplayerbc_r){
          lay = layers(E,k);
          if((!ontop)&&(!onbottom)&&(lv==E->mesh.gridmax))
            ncount++;		/* not in top or bottom */
          if(E->mesh.topvbc != 1) {	/* free slip */
            internal_horizontal_bc(E,E->sphere.cap[1].VB,k,1,0.0,VBX,0,lv);
            if(ontop || onbottom)
              internal_horizontal_bc(E,E->sphere.cap[1].VB,k,3,0.0,VBZ,1,lv);
            internal_horizontal_bc(E,E->sphere.cap[1].VB,k,2,0.0,VBY,0,lv);
            internal_horizontal_bc(E,E->sphere.cap[1].VB,k,1,E->control.VBXtopval,SBX,1,lv);
            if(ontop || onbottom)
              internal_horizontal_bc(E,E->sphere.cap[1].VB,k,3,0.0,SBZ,0,lv);
            internal_horizontal_bc(E,E->sphere.cap[1].VB,k,2,E->control.VBYtopval,SBY,1,lv);
          }else{		/* no slip */
            internal_horizontal_bc(E,E->sphere.cap[1].VB,k,1,E->control.VBXtopval,VBX,1,lv);
            if(ontop || onbottom)
              internal_horizontal_bc(E,E->sphere.cap[1].VB,k,3,0.0,VBZ,1,lv);
            internal_horizontal_bc(E,E->sphere.cap[1].VB,k,2,E->control.VBYtopval,VBY,1,lv);
            internal_horizontal_bc(E,E->sphere.cap[1].VB,k,1,0.0,SBX,0,lv);
            if(ontop || onbottom)
              internal_horizontal_bc(E,E->sphere.cap[1].VB,k,3,0.0,SBZ,0,lv);
            internal_horizontal_bc(E,E->sphere.cap[1].VB,k,2,0.0,SBY,0,lv);
          }
        }
      }
    }
    /* read in velocities/stresses from grd file? */
#ifdef USE_GGRD
    if(E->control.ggrd.vtop_control)
      ggrd_read_vtop_from_file(E, TRUE);
#endif
    /* end toplayerbc > 0 branch */
  }else if(E->mesh.toplayerbc < 0){ 
    /* internal node at noz-toplayerbc */
    for(lv=E->mesh.gridmax;lv>=E->mesh.gridmin;lv--) {
      noz = E->lmesh.NOZ[lv];
      /* we're looping through all nodes for the possibility that
         there are several internal processors which need BCs */
      if(lv == E->mesh.gridmax)
        k = noz + E->mesh.toplayerbc;
      else{
        k = noz + (int)((float)E->mesh.toplayerbc / pow(2.,(float)(E->mesh.gridmax-lv)));
      }
      if(k <= 1)
        myerror(E,"out of bounds for noz and toplayerbc");
      ontop = ((k==noz) && (E->parallel.me_loc[3]==E->parallel.nprocz-1))?(1):(0);
      onbottom = ((k==1) && (E->parallel.me_loc[3]==0))?(1):(0);
      if((!ontop)&&(!onbottom)&&(lv==E->mesh.gridmax))
        ncount++;		/* not in top or bottom */
      if(E->mesh.topvbc != 1) {	/* free slip */
        internal_horizontal_bc(E,E->sphere.cap[1].VB,k,1,0.0,VBX,0,lv);
        if(ontop || onbottom)
          internal_horizontal_bc(E,E->sphere.cap[1].VB,k,3,0.0,VBZ,1,lv);
        internal_horizontal_bc(E,E->sphere.cap[1].VB,k,2,0.0,VBY,0,lv);
        internal_horizontal_bc(E,E->sphere.cap[1].VB,k,1,E->control.VBXtopval,SBX,1,lv);
        if(ontop || onbottom)
          internal_horizontal_bc(E,E->sphere.cap[1].VB,k,3,0.0,SBZ,0,lv);
        internal_horizontal_bc(E,E->sphere.cap[1].VB,k,2,E->control.VBYtopval,SBY,1,lv);
      }else{		/* no slip */
        internal_horizontal_bc(E,E->sphere.cap[1].VB,k,1,E->control.VBXtopval,VBX,1,lv);
        if(ontop || onbottom)
          internal_horizontal_bc(E,E->sphere.cap[1].VB,k,3,0.0,VBZ,1,lv);
        internal_horizontal_bc(E,E->sphere.cap[1].VB,k,2,E->control.VBYtopval,VBY,1,lv);
        internal_horizontal_bc(E,E->sphere.cap[1].VB,k,1,0.0,SBX,0,lv);
        if(ontop || onbottom)
          internal_horizontal_bc(E,E->sphere.cap[1].VB,k,3,0.0,SBZ,0,lv);
        internal_horizontal_bc(E,E->sphere.cap[1].VB,k,2,0.0,SBY,0,lv);
      }
    }
    /* read in velocities/stresses from grd file? */
#ifdef USE_GGRD
    if(E->control.ggrd.vtop_control)
      ggrd_read_vtop_from_file(E, TRUE);
#endif
    /* end toplayerbc < 0 branch */
  }
  if(ncount)
    fprintf(stderr,"assign_internal_bc: CPU %4i (%s): WARNING: assigned internal %s BCs to %6i nodes\n",
	    E->parallel.me,((E->parallel.me_loc[3]==0)&&(E->parallel.nprocz!=1))?("bottom"):
	    ((E->parallel.me_loc[3]==E->parallel.nprocz-1)?("top"):("interior")),
	    (E->mesh.topvbc!=1)?("stress"):("velocity"),ncount);
}
/* End of file  */
