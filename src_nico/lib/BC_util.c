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
#include "math.h"

void read_internal_velocity_from_file(struct All_variables *);
//void get_bcs_id_for_residual();
void construct_id();

/* XXX DJB */
void internal_velocity_bc(E)
     struct All_variables *E;
{    int i,j,k,m,level,nodel,node;
     int nox1,noy1,noz1;
     int nox,noy,noz,levdepth;
     // float theta,phi,r;

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
    // fprintf(stderr,"Internal Velocity: Setting VBX, VBY, VBZ flags\n");
  }

  /* for debugging */
  // fprintf(stderr,"gridmax=%d, gridmin=%d, levmax=%d, levmin=%d\n",E->mesh.gridmax,E->mesh.gridmin,E->mesh.levmax,E->mesh.levmin);

  nox = E->lmesh.nox;
  noy = E->lmesh.noy;
  noz = E->lmesh.noz;

  // fprintf(stderr,"Internal Velocity: nox=%d, noy=%d, noz=%d\n",nox,noy,noz);

  /* update VBX, VBY, VBZ, SBX, SBY, SBZ flags (at all levels) */
  for(level=E->mesh.levmax;level>=E->mesh.levmin;level--) {

    /* for debugging */
    // fprintf(stderr,"level=%d, levdepth=%d, nox1=%d, noy1=%d, noz1=%d\n",level,levdepth,nox1,noy1,noz1);
    // fprintf(stderr,"level=%d, levdepth=%d, nox=%d, noy=%d, noz=%d\n",level,levdepth,nox,noy,noz);

    // XXX DJB - OPTION 1 - prescribe bcs at all levels
    if ( (E->control.CONJ_GRAD && level==E->mesh.levmax) || E->control.NMULTIGRID)  {

    // XXX DJB - OPTION 2 - let's try just prescribing at finest mesh
    //if ( level==E->mesh.levmax )  {

      /* `depth' in multigrid */
      levdepth = E->mesh.levmax-level;

      nox1 = E->lmesh.NOX[level];
      noy1 = E->lmesh.NOY[level];
      noz1 = E->lmesh.NOZ[level];

      /* for debugging */
      //fprintf(stderr,"level=%d, levdepth=%d, nox=%d, noy=%d, noz=%d\n",level,levdepth,nox,noy,noz);
      // fprintf(stderr,"level=%d, levdepth=%d, nox1=%d, noy1=%d, noz1=%d\n",level,levdepth,nox1,noy1,noz1);

      for(m=1;m<=E->sphere.caps_per_proc;m++) {
        for(j=1;j<=noy1;j++)
          for(i=1;i<=nox1;i++)
            for(k=1;k<=noz1;k++) {
              nodel = k+(i-1)*noz1+(j-1)*noz1*nox1;
              node = 1 + (k-1)*pow(2,levdepth); // for k
              node += (i-1)*noz*pow(2,levdepth); // for i
              node += (j-1)*noz*nox*pow(2,levdepth); // for j

              /* XXX DJB - debugging */
              /* theta = E->SX[level][m][1][nodel];
              phi = E->SX[level][m][2][nodel];
              r = E->SX[level][m][3][nodel]; */

              /* XXX DJB - debugging */
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
                // fprintf(stderr,"Internal Velocity: level=%d, levdepth=%d, m=%d, nodel=%d, node=%d\n",level,levdepth,m,nodel,node);
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
        //fprintf(stderr,"Internal Velocity: Update E->zero_resid\n"))
        // get_bcs_id_for_residual(E,level,m);

      } /* end loop over m */
    } /* end if */
  } /* end loop over level */

  /* suggested by Eh Tan, presumably rather than get_bcs_id_for_residual */
  construct_id(E);

  // fprintf(stderr,"DONE with internal_velocity_bc\n");
 
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


/* End of file  */

