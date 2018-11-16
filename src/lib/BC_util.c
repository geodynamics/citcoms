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
#include "global_defs.h"
void horizontal_bc(struct All_variables *,float *[],int,int,float,unsigned int,char,int,int);
void internal_horizontal_bc(struct All_variables *,float *[],int,int,float,unsigned int,char,int,int);
void myerror(struct All_variables *,char *);
int layers(struct All_variables *,int,int);
void read_internal_velocity_from_file(struct All_variables *); // DJB SLAB
//void get_bcs_id_for_residual(); // DJB SLAB
void construct_id(); // DJB SLAB

#ifdef USE_GGRD
#include "ggrd_handling.h"
#endif

/* DJB SLAB */
void internal_velocity_bc(E)
     struct All_variables *E;
{    int i,j,k,m,level,nodel,node;
     int nox1,noy1,noz1;
     int nox,noy,noz,levdepth;
     /* float theta,phi,r; */

  /* read internal velocity from file */
  /* nodes are ignored with flag !=0 or !=1 */
  read_internal_velocity_from_file(E);

  if(E->control.verbose) {
      fprintf(E->fp_out,"INSIDE internal_velocity_bc\n");
      fflush(E->fp_out);
  }

  if(E->parallel.me==0){
      fprintf(E->fp,"Internal Velocity: Setting VBX, VBY, VBZ flags\n");
      fflush(E->fp);
    /* fprintf(stderr,"Internal Velocity: Setting VBX, VBY, VBZ flags\n"); */
  }

  /* for debugging */
  /* fprintf(stderr,"gridmax=%d, gridmin=%d, levmax=%d, levmin=%d\n",E->mesh.gridmax,E->mesh.gridmin,E->mesh.levmax,E->mesh.levmin); */

  nox = E->lmesh.nox;
  noy = E->lmesh.noy;
  noz = E->lmesh.noz;

  /* fprintf(stderr,"Internal Velocity: nox=%d, noy=%d, noz=%d\n",nox,noy,noz); */

  /* update VBX, VBY, VBZ, SBX, SBY, SBZ flags (at all levels) */
  for(level=E->mesh.levmax;level>=E->mesh.levmin;level--) {

    /* for debugging */
    /* fprintf(stderr,"level=%d, levdepth=%d, nox1=%d, noy1=%d, noz1=%d\n",level,levdepth,nox1,noy1,noz1); */
    /* fprintf(stderr,"level=%d, levdepth=%d, nox=%d, noy=%d, noz=%d\n",level,levdepth,nox,noy,noz); */

    /* DJB SLAB - prescribe internal velocity boundary conditions at all levels.  This is the recommended
       approach, see Issue #6 in the bitbucket repo */
    if ( (E->control.CONJ_GRAD && level==E->mesh.levmax) || E->control.NMULTIGRID)  {

    /* DJB SLAB - prescribe at finest mesh (done for testing) */
    /*if ( level==E->mesh.levmax )  { */

      /* this is the depth in multigrid */
      levdepth = E->mesh.levmax-level;

      nox1 = E->lmesh.NOX[level];
      noy1 = E->lmesh.NOY[level];
      noz1 = E->lmesh.NOZ[level];

      /* for debugging */
      /*fprintf(stderr,"level=%d, levdepth=%d, nox=%d, noy=%d, noz=%d\n",level,levdepth,nox,noy,noz); */
      /* fprintf(stderr,"level=%d, levdepth=%d, nox1=%d, noy1=%d, noz1=%d\n",level,levdepth,nox1,noy1,noz1); */

      for(m=1;m<=E->sphere.caps_per_proc;m++) {
        for(j=1;j<=noy1;j++)
          for(i=1;i<=nox1;i++)
            for(k=1;k<=noz1;k++) {
              nodel = k+(i-1)*noz1+(j-1)*noz1*nox1;
              node = 1 + (k-1)*pow(2,levdepth); // for k
              node += (i-1)*noz*pow(2,levdepth); // for i
              node += (j-1)*noz*nox*pow(2,levdepth); // for j

              /* DJB - debugging */
              /* theta = E->SX[level][m][1][nodel];
              phi = E->SX[level][m][2][nodel];
              r = E->SX[level][m][3][nodel]; */

              /* DJB - debugging */
              /* if(E->sphere.cap[m].slab_sten2[node]==1) {
                fprintf(stderr,"%d %d %d %d %d %d %d %d %d %d %f %f %f %d\n",level,m,noy1,nox1,noz1,j,i,k,nodel,node,theta,phi,r,E->sphere.cap[m].slab_sten2[node]);
              } */

              if(E->sphere.cap[m].slab_sten2[node]==0) { // turn OFF all velocity and stress BC's
                E->NODE[level][m][nodel] = E->NODE[level][m][nodel] & (~VBX);
                E->NODE[level][m][nodel] = E->NODE[level][m][nodel] & (~VBY);
                E->NODE[level][m][nodel] = E->NODE[level][m][nodel] & (~VBZ);
                E->NODE[level][m][nodel] = E->NODE[level][m][nodel] & (~SBX);
                E->NODE[level][m][nodel] = E->NODE[level][m][nodel] & (~SBY);
                E->NODE[level][m][nodel] = E->NODE[level][m][nodel] & (~SBZ);
              }
              else if(E->sphere.cap[m].slab_sten2[node]==1) { // turn ON velocity BC's

                if(E->control.verbose) {
                  fprintf(E->fp_out,"Apply internal velocity boundary condition\n");
                  fflush(E->fp_out);
                }
                /* for debugging */
                /* fprintf(stderr,"Internal Velocity: level=%d, levdepth=%d, m=%d, nodel=%d, node=%d\n",level,levdepth,m,nodel,node); */
                E->NODE[level][m][nodel] = E->NODE[level][m][nodel] | (VBX);
                E->NODE[level][m][nodel] = E->NODE[level][m][nodel] | (VBY);
                E->NODE[level][m][nodel] = E->NODE[level][m][nodel] | (VBZ);
                E->NODE[level][m][nodel] = E->NODE[level][m][nodel] & (~SBX);
                E->NODE[level][m][nodel] = E->NODE[level][m][nodel] & (~SBY);
                E->NODE[level][m][nodel] = E->NODE[level][m][nodel] & (~SBZ);
              }
           /* else */
             /* do nothing, leave BC's untouched */

        } /* end i, j, k loop */

        /* this is now done in construct_id */
        /* update E->zero_resid which is used by solvers */
        /*fprintf(stderr,"Internal Velocity: Update E->zero_resid\n")) */
        /* get_bcs_id_for_residual(E,level,m); */

      } /* end loop over m */
    } /* end if */
  } /* end loop over level */

  /* suggested by Eh Tan, presumably rather than get_bcs_id_for_residual */
  construct_id(E);

  /* fprintf(stderr,"DONE with internal_velocity_bc\n"); */

  return;
}

/* 

assign boundary conditions to a horizontal layer of nodes within mesh,
without consideration of being in top or bottom processor

*/
void internal_horizontal_bc(struct All_variables *E,float *BC[],int row,int dirn,
			    float value,unsigned int mask,char onoff,int level,int m)
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
  if (onoff == 0)          {
    for(j=1;j<=E->lmesh.NOY[level];j++)
      for(i=1;i<=E->lmesh.NOX[level];i++)     {
	node = row+(i-1)*E->lmesh.NOZ[level]+(j-1)*noxnoz;
	E->NODE[level][m][node] = E->NODE[level][m][node] & (~ mask);
      }        /* end for loop i & j */
  }else        {
    /* turn bc marker to one */
    for(j=1;j<=E->lmesh.NOY[level];j++)
      for(i=1;i<=E->lmesh.NOX[level];i++)       {
	node = row+(i-1)*E->lmesh.NOZ[level]+(j-1)*noxnoz;
	E->NODE[level][m][node] = E->NODE[level][m][node] | (mask);
	if(level == E->mesh.levmax)   /* NB */
	  BC[dirn][node] = value;
      }     /* end for loop i & j */
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
    for(lv=E->mesh.gridmax;lv>=E->mesh.gridmin;lv--)
      for (j=1;j<=E->sphere.caps_per_proc;j++)     {
	noz = E->lmesh.NOZ[lv];
	/* we're looping through all nodes for the possibility that
	   there are several internal processors which need BCs */
	for(k=noz;k >= 1;k--){ /* assumes regular grid */
	  ontop    = ((k==noz) && (E->parallel.me_loc[3]==E->parallel.nprocz-1))?(1):(0);
	  onbottom = ((k==1) && (E->parallel.me_loc[3]==0))?(1):(0);
	  /* node number is k, assuming no dependence on x and y  */
	  if(E->SX[lv][j][3][k] >= E->mesh.toplayerbc_r){
	    lay = layers(E,j,k);
	    if((!ontop)&&(!onbottom)&&(lv==E->mesh.gridmax))
	      ncount++;		/* not in top or bottom */
	    if(E->mesh.topvbc != 1) {	/* free slip */
	      internal_horizontal_bc(E,E->sphere.cap[j].VB,k,1,0.0,VBX,0,lv,j);
	      if(ontop || onbottom)
		internal_horizontal_bc(E,E->sphere.cap[j].VB,k,3,0.0,VBZ,1,lv,j);
	      internal_horizontal_bc(E,E->sphere.cap[j].VB,k,2,0.0,VBY,0,lv,j);
	      internal_horizontal_bc(E,E->sphere.cap[j].VB,k,1,E->control.VBXtopval,SBX,1,lv,j);
	      if(ontop || onbottom)
		internal_horizontal_bc(E,E->sphere.cap[j].VB,k,3,0.0,SBZ,0,lv,j);
	      internal_horizontal_bc(E,E->sphere.cap[j].VB,k,2,E->control.VBYtopval,SBY,1,lv,j);
	    }else{		/* no slip */
	      internal_horizontal_bc(E,E->sphere.cap[j].VB,k,1,E->control.VBXtopval,VBX,1,lv,j);
	      if(ontop || onbottom)
		internal_horizontal_bc(E,E->sphere.cap[j].VB,k,3,0.0,VBZ,1,lv,j);
	      internal_horizontal_bc(E,E->sphere.cap[j].VB,k,2,E->control.VBYtopval,VBY,1,lv,j);
	      internal_horizontal_bc(E,E->sphere.cap[j].VB,k,1,0.0,                 SBX,0,lv,j);
	      if(ontop || onbottom)
		internal_horizontal_bc(E,E->sphere.cap[j].VB,k,3,0.0,SBZ,0,lv,j);
	      internal_horizontal_bc(E,E->sphere.cap[j].VB,k,2,0.0,                 SBY,0,lv,j);
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
    for(lv=E->mesh.gridmax;lv>=E->mesh.gridmin;lv--)
      for (j=1;j<=E->sphere.caps_per_proc;j++)     {
	noz = E->lmesh.NOZ[lv];
	/* we're looping through all nodes for the possibility that
	   there are several internal processors which need BCs */
	if(lv == E->mesh.gridmax)
	  k = noz + E->mesh.toplayerbc;
	else{
	  k = noz + (int)((float)E->mesh.toplayerbc / pow(2.,(float)(E->mesh.gridmax-lv)));
	}
	//fprintf(stderr,"BC_util: inner node: CPU: %i lv %i noz %i k %i\n",E->parallel.me,lv,noz,k);
	if(k <= 1)
	  myerror(E,"out of bounds for noz and toplayerbc");
	ontop    = ((k==noz) && (E->parallel.me_loc[3]==E->parallel.nprocz-1))?(1):(0);
	onbottom = ((k==1) && (E->parallel.me_loc[3]==0))?(1):(0);
	if((!ontop)&&(!onbottom)&&(lv==E->mesh.gridmax))
	  ncount++;		/* not in top or bottom */
	if(E->mesh.topvbc != 1) {	/* free slip */
	  internal_horizontal_bc(E,E->sphere.cap[j].VB,k,1,0.0,VBX,0,lv,j);
	  if(ontop || onbottom)
	    internal_horizontal_bc(E,E->sphere.cap[j].VB,k,3,0.0,VBZ,1,lv,j);
	  internal_horizontal_bc(E,E->sphere.cap[j].VB,k,2,0.0,VBY,0,lv,j);
	  internal_horizontal_bc(E,E->sphere.cap[j].VB,k,1,E->control.VBXtopval,SBX,1,lv,j);
	  if(ontop || onbottom)
	    internal_horizontal_bc(E,E->sphere.cap[j].VB,k,3,0.0,SBZ,0,lv,j);
	  internal_horizontal_bc(E,E->sphere.cap[j].VB,k,2,E->control.VBYtopval,SBY,1,lv,j);
	}else{		/* no slip */
	  internal_horizontal_bc(E,E->sphere.cap[j].VB,k,1,E->control.VBXtopval,VBX,1,lv,j);
	  if(ontop || onbottom)
	    internal_horizontal_bc(E,E->sphere.cap[j].VB,k,3,0.0,VBZ,1,lv,j);
	  internal_horizontal_bc(E,E->sphere.cap[j].VB,k,2,E->control.VBYtopval,VBY,1,lv,j);
	  internal_horizontal_bc(E,E->sphere.cap[j].VB,k,1,0.0,                 SBX,0,lv,j);
	  if(ontop || onbottom)
	    internal_horizontal_bc(E,E->sphere.cap[j].VB,k,3,0.0,SBZ,0,lv,j);
	  internal_horizontal_bc(E,E->sphere.cap[j].VB,k,2,0.0,                 SBY,0,lv,j);
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

