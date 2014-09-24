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
#include <math.h>

#include "element_definitions.h"
#include "global_defs.h"
#include "sphere_communication.h"

#include "parallel_related.h"


static void set_horizontal_communicator(struct All_variables*);
static void set_vertical_communicator(struct All_variables*);

static void exchange_node_d(struct All_variables *, double*, int);
static void exchange_node_f(struct All_variables *, float*, int);


/* ============================================ */
/* ============================================ */

void regional_parallel_processor_setup(struct All_variables *E)
  {

  int i,j,k,m,me,temp,pid_surf;
  int surf_proc_per_cap, proc_per_cap, total_proc;

  me = E->parallel.me;

  surf_proc_per_cap = E->parallel.nprocx * E->parallel.nprocy;
  proc_per_cap = surf_proc_per_cap * E->parallel.nprocz;
  total_proc = E->sphere.caps * proc_per_cap;
  E->parallel.total_surf_proc = E->sphere.caps * surf_proc_per_cap;

  if ( total_proc != E->parallel.nproc ) {
    if (E->parallel.me==0) fprintf(stderr,"!!!! # of requested CPU is incorrect \n");
    parallel_process_termination();
    }

  /* determine the location of processors in each cap */
  /* z direction first */
  j = me % E->parallel.nprocz;
  E->parallel.me_loc[3] = j;

  /* x direction then */
  k = (me - j)/E->parallel.nprocz % E->parallel.nprocx;
  E->parallel.me_loc[1] = k;

  /* y direction then */
  i = ((me - j)/E->parallel.nprocz - k)/E->parallel.nprocx % E->parallel.nprocy;
  E->parallel.me_loc[2] = i;

  E->sphere.caps_per_proc = 1;

  /* determine cap id for each cap in a given processor  */
  pid_surf = me/E->parallel.nprocz;
  i = cases[E->sphere.caps_per_proc];

  E->sphere.capid[CPPR] = 1;

  /* steup location-to-processor map */
  E->parallel.loc2proc_map = (int ****) malloc(E->sphere.caps*sizeof(int ***));
  for (m=0;m<E->sphere.caps;m++)  {
    E->parallel.loc2proc_map[m] = (int ***) malloc(E->parallel.nprocx*sizeof(int **));
    for (i=0;i<E->parallel.nprocx;i++) {
      E->parallel.loc2proc_map[m][i] = (int **) malloc(E->parallel.nprocy*sizeof(int *));
      for (j=0;j<E->parallel.nprocy;j++)
	E->parallel.loc2proc_map[m][i][j] = (int *) malloc(E->parallel.nprocz*sizeof(int));
    }
  }

  for (m=0;m<E->sphere.caps;m++)
    for (i=0;i<E->parallel.nprocx;i++)
      for (j=0;j<E->parallel.nprocy;j++)
	for (k=0;k<E->parallel.nprocz;k++) {
	    E->parallel.loc2proc_map[m][i][j][k] = m*proc_per_cap
	      + j*E->parallel.nprocx*E->parallel.nprocz
	      + i*E->parallel.nprocz + k;
	}

  if (E->control.verbose) {
    fprintf(E->fp_out,"me=%d loc1=%d loc2=%d loc3=%d\n",me,E->parallel.me_loc[1],E->parallel.me_loc[2],E->parallel.me_loc[3]);
    fprintf(E->fp_out,"capid[%d]=%d \n",CPPR,E->sphere.capid[CPPR]);
    for (m=0;m<E->sphere.caps;m++)
      for (j=0;j<E->parallel.nprocy;j++)
	for (i=0;i<E->parallel.nprocx;i++)
	  for (k=0;k<E->parallel.nprocz;k++)
	    fprintf(E->fp_out,"loc2proc_map[cap=%d][x=%d][y=%d][z=%d] = %d\n",
		    m,i,j,k,E->parallel.loc2proc_map[m][i][j][k]);

    fflush(E->fp_out);
  }

  set_horizontal_communicator(E);
  set_vertical_communicator(E);

  E->exchange_node_d = exchange_node_d;
  E->exchange_node_f = exchange_node_f;

}


static void set_horizontal_communicator(struct All_variables *E)
{
  MPI_Group world_g, horizon_g;
  int i,j,k,m,n;
  int *processors;

  processors = (int *) malloc((E->parallel.total_surf_proc+1)*sizeof(int));

  k = E->parallel.me_loc[3];
  n = 0;
  for (m=0;m<E->sphere.caps;m++)
    for (i=0;i<E->parallel.nprocx;i++)
      for (j=0;j<E->parallel.nprocy;j++) {
	processors[n] = E->parallel.loc2proc_map[m][i][j][k];
	n++;
      }

  if (E->control.verbose) {
    fprintf(E->fp_out,"horizontal group of me=%d loc3=%d\n",E->parallel.me,E->parallel.me_loc[3]);
    for (j=0;j<E->parallel.total_surf_proc;j++) {
      fprintf(E->fp_out,"%d proc=%d\n",j,processors[j]);
    }
    fflush(E->fp_out);
  }

  MPI_Comm_group(E->parallel.world, &world_g);
  MPI_Group_incl(world_g, E->parallel.total_surf_proc, processors, &horizon_g);
  MPI_Comm_create(E->parallel.world, horizon_g, &(E->parallel.horizontal_comm));


  MPI_Group_free(&horizon_g);
  MPI_Group_free(&world_g);
  free((void *) processors);

}


static void set_vertical_communicator(struct All_variables *E)
{
  MPI_Group world_g, vertical_g;
  int i,j,k,m;
  int *processors;

  processors = (int *)malloc((E->parallel.nprocz+2)*sizeof(int));
  if (!processors)
    fprintf(stderr,"no memory!!\n");

  m = E->sphere.capid[1] - 1;  /* assume 1 cap per proc. */
  i = E->parallel.me_loc[1];
  j = E->parallel.me_loc[2];

  for (k=0;k<E->parallel.nprocz;k++) {
      processors[k] = E->parallel.loc2proc_map[m][i][j][k];
  }

  if (E->control.verbose) {
    fprintf(E->fp_out,"vertical group of me=%d loc1=%d loc2=%d\n",E->parallel.me,E->parallel.me_loc[1],E->parallel.me_loc[2]);
    for (k=0;k<E->parallel.nprocz;k++) {
      fprintf(E->fp_out,"%d proc=%d\n",k,processors[k]);
    }
    fflush(E->fp_out);
  }

  MPI_Comm_group(E->parallel.world, &world_g);
  MPI_Group_incl(world_g, E->parallel.nprocz, processors, &vertical_g);
  MPI_Comm_create(E->parallel.world, vertical_g, &(E->parallel.vertical_comm));

  MPI_Group_free(&vertical_g);
  MPI_Group_free(&world_g);
  free((void *) processors);
}



/* =========================================================================
get element information for each processor.
 ========================================================================= */

void regional_parallel_domain_decomp0(struct All_variables *E)
  {

  int i,nox,noz,noy,me;

  me = E->parallel.me;

  E->lmesh.elx = E->mesh.elx/E->parallel.nprocx;
  E->lmesh.elz = E->mesh.elz/E->parallel.nprocz;
  E->lmesh.ely = E->mesh.ely/E->parallel.nprocy;
  E->lmesh.nox = E->lmesh.elx + 1;
  E->lmesh.noz = E->lmesh.elz + 1;
  E->lmesh.noy = E->lmesh.ely + 1;

  E->lmesh.exs = E->parallel.me_loc[1]*E->lmesh.elx;
  E->lmesh.eys = E->parallel.me_loc[2]*E->lmesh.ely;
  E->lmesh.ezs = E->parallel.me_loc[3]*E->lmesh.elz;
  E->lmesh.nxs = E->parallel.me_loc[1]*E->lmesh.elx+1;
  E->lmesh.nys = E->parallel.me_loc[2]*E->lmesh.ely+1;
  E->lmesh.nzs = E->parallel.me_loc[3]*E->lmesh.elz+1;

  E->lmesh.nno = E->lmesh.noz*E->lmesh.nox*E->lmesh.noy;
  E->lmesh.nel = E->lmesh.ely*E->lmesh.elx*E->lmesh.elz;
  E->lmesh.npno = E->lmesh.nel;

  E->lmesh.nsf = E->lmesh.nno/E->lmesh.noz;
  E->lmesh.snel = E->lmesh.elx*E->lmesh.ely;


  for(i=E->mesh.levmax;i>=E->mesh.levmin;i--)   {

     if (E->control.NMULTIGRID)  {
        nox = E->mesh.mgunitx * (int) pow(2.0,(double)i) + 1;
        noy = E->mesh.mgunity * (int) pow(2.0,(double)i) + 1;
        noz = E->lmesh.elz/(int)pow(2.0,(double)(E->mesh.levmax-i))+1;
        E->parallel.redundant[i]=0;
        }
     else
        { noz = E->lmesh.noz;
          noy = E->mesh.mgunity * (int) pow(2.0,(double)i) + 1;
          nox = E->mesh.mgunitx * (int) pow(2.0,(double)i) + 1;
          if(i<E->mesh.levmax) noz=2;
        }

     E->lmesh.ELX[i] = nox-1;
     E->lmesh.ELY[i] = noy-1;
     E->lmesh.ELZ[i] = noz-1;
     E->lmesh.NOZ[i] = noz;
     E->lmesh.NOY[i] = noy;
     E->lmesh.NOX[i] = nox;
     E->lmesh.NNO[i] = nox * noz * noy;
     E->lmesh.NNOV[i] = E->lmesh.NNO[i];
     E->lmesh.SNEL[i] = E->lmesh.ELX[i]*E->lmesh.ELY[i];

     E->lmesh.NEL[i] = (nox-1) * (noz-1) * (noy-1);
     E->lmesh.NPNO[i] = E->lmesh.NEL[i] ;

     E->lmesh.NEQ[i] = E->mesh.nsd * E->lmesh.NNOV[i] ;

     E->lmesh.EXS[i] = E->parallel.me_loc[1]*E->lmesh.ELX[i];
     E->lmesh.EYS[i] = E->parallel.me_loc[2]*E->lmesh.ELY[i];
     E->lmesh.EZS[i] = E->parallel.me_loc[3]*E->lmesh.ELZ[i];
     E->lmesh.NXS[i] = E->parallel.me_loc[1]*E->lmesh.ELX[i]+1;
     E->lmesh.NYS[i] = E->parallel.me_loc[2]*E->lmesh.ELY[i]+1;
     E->lmesh.NZS[i] = E->parallel.me_loc[3]*E->lmesh.ELZ[i]+1;

     }

/*
fprintf(stderr,"b %d %d %d %d %d %d %d\n",E->parallel.me,E->parallel.me_loc[1],E->parallel.me_loc[2],E->parallel.me_loc[3],E->lmesh.nzs,E->lmesh.nys,E->lmesh.noy);
*/
/* parallel_process_termination();
*/
}




/* ============================================
 determine boundary nodes for
 exchange info across the boundaries
 ============================================ */

void regional_parallel_domain_boundary_nodes(E)
  struct All_variables *E;
  {

  void parallel_process_termination();

  int m,i,ii,j,k,l,node,el,lnode;
  int lev,ele,elx,elz,ely,nel,nno,nox,noz,noy;
  FILE *fp,*fp1;
  char output_file[255];

  for(lev=E->mesh.gridmin;lev<=E->mesh.gridmax;lev++)   {
      nel = E->lmesh.NEL[lev];
      elx = E->lmesh.ELX[lev];
      elz = E->lmesh.ELZ[lev];
      ely = E->lmesh.ELY[lev];
      nox = E->lmesh.NOX[lev];
      noy = E->lmesh.NOY[lev];
      noz = E->lmesh.NOZ[lev];
      nno = E->lmesh.NNO[lev];

/* do the ZOY boundary elements first */
      lnode = 0;
      ii =1;              /* left */
      for(j=1;j<=noz;j++)
      for(k=1;k<=noy;k++)  {
        node = j + (k-1)*noz*nox;
        E->parallel.NODE[lev][CPPR][++lnode].bound[ii] =  node;
        E->NODE[lev][node] = E->NODE[lev][node] | OFFSIDE;
        }

      E->parallel.NUM_NNO[lev].bound[ii] = lnode;


      lnode = 0;
      ii =2;              /* right */
      for(j=1;j<=noz;j++)
      for(k=1;k<=noy;k++)      {
        node = (nox-1)*noz + j + (k-1)*noz*nox;
        E->parallel.NODE[lev][CPPR][++lnode].bound[ii] =  node;
        E->NODE[lev][node] = E->NODE[lev][node] | OFFSIDE;
        }

      E->parallel.NUM_NNO[lev].bound[ii] = lnode;


/* do XOY boundary elements */
      ii=5;                           /* bottom */
      lnode=0;
      for(k=1;k<=noy;k++)
      for(i=1;i<=nox;i++)   {
        node = (k-1)*nox*noz + (i-1)*noz + 1;
        E->parallel.NODE[lev][CPPR][++lnode].bound[ii] = node;
        E->NODE[lev][node] = E->NODE[lev][node] | OFFSIDE;
        }

      E->parallel.NUM_NNO[lev].bound[ii] = lnode;

      ii=6;                           /* top  */
      lnode=0;
      for(k=1;k<=noy;k++)
      for(i=1;i<=nox;i++)  {
        node = (k-1)*nox*noz + i*noz;
        E->parallel.NODE[lev][CPPR][++lnode].bound[ii] = node;
        E->NODE[lev][node] = E->NODE[lev][node] | OFFSIDE;
        }

      E->parallel.NUM_NNO[lev].bound[ii] = lnode;


/* do XOZ boundary elements for 3D */
      ii=3;                           /* front */
      lnode=0;
      for(j=1;j<=noz;j++)
      for(i=1;i<=nox;i++)   {
        node = (i-1)*noz +j;
        E->parallel.NODE[lev][CPPR][++lnode].bound[ii] = node;
        E->NODE[lev][node] = E->NODE[lev][node] | OFFSIDE;
        }

      E->parallel.NUM_NNO[lev].bound[ii] = lnode;

      ii=4;                           /* rear */
      lnode=0;
      for(j=1;j<=noz;j++)
      for(i=1;i<=nox;i++)   {
        node = noz*nox*(noy-1) + (i-1)*noz +j;
        E->parallel.NODE[lev][CPPR][++lnode].bound[ii] = node;
        E->NODE[lev][node] = E->NODE[lev][node] | OFFSIDE;
        }

      E->parallel.NUM_NNO[lev].bound[ii] = lnode;

         /* determine the overlapped nodes between caps or between proc */

    if (E->parallel.me_loc[1]!=E->parallel.nprocx-1)
      for (lnode=1;lnode<=E->parallel.NUM_NNO[lev].bound[2];lnode++) {
        node = E->parallel.NODE[lev][CPPR][lnode].bound[2];
        E->NODE[lev][node] = E->NODE[lev][node] | SKIP;
        }

    if (E->parallel.me_loc[2]!=E->parallel.nprocy-1)
      for (lnode=1;lnode<=E->parallel.NUM_NNO[lev].bound[4];lnode++) {
        node = E->parallel.NODE[lev][CPPR][lnode].bound[4];
        E->NODE[lev][node] = E->NODE[lev][node] | SKIP;
        }

    if (E->parallel.me_loc[3]!=E->parallel.nprocz-1)
      for (lnode=1;lnode<=E->parallel.NUM_NNO[lev].bound[6];lnode++) {
        node = E->parallel.NODE[lev][CPPR][lnode].bound[6];
        E->NODE[lev][node] = E->NODE[lev][node] | SKIP;
        }

    }   /* end for level */


if (E->control.verbose) {
 fprintf(E->fp_out,"output_shared_nodes %d \n",E->parallel.me);
 for(lev=E->mesh.gridmax;lev>=E->mesh.gridmin;lev--)
    fprintf(E->fp_out,"lev=%d  me=%d capid=%d m=%d \n",lev,E->parallel.me,E->sphere.capid[CPPR],CPPR);
    for (ii=1;ii<=6;ii++)
      for (i=1;i<=E->parallel.NUM_NNO[lev].bound[ii];i++)
        fprintf(E->fp_out,"ii=%d   %d %d \n",ii,i,E->parallel.NODE[lev][CPPR][i].bound[ii]);

    lnode=0;
    for (node=1;node<=E->lmesh.NNO[lev];node++)
      if((E->NODE[lev][node] & SKIP)) {
        lnode++;
        fprintf(E->fp_out,"skip %d %d \n",lnode,node);
        }
 fflush(E->fp_out);
 }
}


/* ============================================
 determine communication routs  and boundary ID for
 exchange info across the boundaries
 assuming fault nodes are in the top row of processors
 ============================================ */

void regional_parallel_communication_routs_v(E)
  struct All_variables *E;
  {

  int m,i,ii,j,k,l,node,el,elt,lnode,jj,doff,target_cap;
  int lev,elx,elz,ely,nno,nox,noz,noy,kkk,kk,kf,kkkp;
  int me, nproczl,nprocxl,nprocyl;
  int temp_dims,addi_doff;
  int cap,lx,ly,lz,dir;
  FILE *fp,*fp1,*fp2;
  char output_file[255];

  const int dims=E->mesh.nsd;

  me = E->parallel.me;
  nproczl = E->parallel.nprocz;
  nprocyl = E->parallel.nprocy;
  nprocxl = E->parallel.nprocx;
  lx = E->parallel.me_loc[1];
  ly = E->parallel.me_loc[2];
  lz = E->parallel.me_loc[3];

        /* determine the communications in horizontal direction        */
  for(lev=E->mesh.gridmax;lev>=E->mesh.gridmin;lev--)       {
    nox = E->lmesh.NOX[lev];
    noz = E->lmesh.NOZ[lev];
    noy = E->lmesh.NOY[lev];
    ii=0;
    kkk=0;


      cap = E->sphere.capid[CPPR] - 1;  /* which cap I am in (0~11) */

          for(i=1;i<=2;i++)       {       /* do YOZ boundaries & OY lines */

        ii ++;
        E->parallel.NUM_PASS[lev].bound[ii] = 1;
        if(E->parallel.me_loc[1]==0 && i==1)
          E->parallel.NUM_PASS[lev].bound[ii] = 0;
        else if(E->parallel.me_loc[1]==nprocxl-1 && i==2)
          E->parallel.NUM_PASS[lev].bound[ii] = 0;

        if (E->parallel.NUM_PASS[lev].bound[ii] == 1)  {
          kkk ++;
              /* determine the pass ID for ii-th boundary and kkk-th pass */

          /*E->parallel.PROCESSOR[lev][m].pass[kkk]=me-((i==1)?1:-1)*nproczl; */
	  dir = ( (i==1)? 1 : -1);
          E->parallel.PROCESSOR[lev][CPPR].pass[kkk]=E->parallel.loc2proc_map[cap][lx-dir][ly][lz];

              E->parallel.NUM_NODE[lev][CPPR].pass[kkk] = E->parallel.NUM_NNO[lev].bound[ii];
          jj = 0;
          for (k=1;k<=E->parallel.NUM_NODE[lev][CPPR].pass[kkk];k++)   {
            lnode = k;
            node = E->parallel.NODE[lev][CPPR][lnode].bound[ii];
            E->parallel.EXCHANGE_NODE[lev][CPPR][k].pass[kkk] = node;
            temp_dims = dims;

                    for(doff=1;doff<=temp_dims;doff++)
                         E->parallel.EXCHANGE_ID[lev][CPPR][++jj].pass[kkk] = E->ID[lev][node].doff[doff];
            }  /* end for node k */

              E->parallel.NUM_NEQ[lev].pass[kkk] = jj;

          }   /* end if */
            }  /* end for i */


        for(k=1;k<=2;k++)        {      /* do XOZ boundaries & OZ lines */
        ii ++;
        E->parallel.NUM_PASS[lev].bound[ii] = 1;
        if(E->parallel.me_loc[2]==0 && k==1)
          E->parallel.NUM_PASS[lev].bound[ii] = 0;
        else if(E->parallel.me_loc[2]==nprocyl-1 && k==2)
          E->parallel.NUM_PASS[lev].bound[ii] = 0;

        if(E->parallel.NUM_PASS[lev].bound[ii] == 1)  {

          kkk ++;
              /* determine the pass ID for ii-th boundary and kkk-th pass */

          /*E->parallel.PROCESSOR[lev][m].pass[kkk]=me-((k==1)?1:-1)*nprocxl*nproczl; */
	  dir = ( (k==1)? 1 : -1);
          E->parallel.PROCESSOR[lev][CPPR].pass[kkk]=E->parallel.loc2proc_map[cap][lx][ly-dir][lz];

          E->parallel.NUM_NODE[lev][CPPR].pass[kkk] = E->parallel.NUM_NNO[lev].bound[ii];

          jj = 0; kf = 0;
          for (kk=1;kk<=E->parallel.NUM_NODE[lev][CPPR].pass[kkk];kk++)   {
            lnode = kk;
            node = E->parallel.NODE[lev][CPPR][lnode].bound[ii];
            E->parallel.EXCHANGE_NODE[lev][CPPR][kk].pass[kkk] = node;
            temp_dims = dims;
                    for(doff=1;doff<=temp_dims;doff++)
                         E->parallel.EXCHANGE_ID[lev][CPPR][++jj].pass[kkk] = E->ID[lev][node].doff[doff];
            }  /* end for node kk */

              E->parallel.NUM_NEQ[lev].pass[kkk] = jj;

          }   /* end if */

            }  /* end for k */


        for(j=1;j<=2;j++)       {       /* do XOY boundaries & OX lines */
        ii ++;
        E->parallel.NUM_PASS[lev].bound[ii] = 1;
        if(E->parallel.me_loc[3]==0 && j==1)
          E->parallel.NUM_PASS[lev].bound[ii] = 0;
        else if(E->parallel.me_loc[3]==nproczl-1 && j==2)
          E->parallel.NUM_PASS[lev].bound[ii] = 0;

        if(E->parallel.NUM_PASS[lev].bound[ii] == 1)  {
          kkk ++;
              /* determine the pass ID for ii-th boundary and kkk-th pass */

          /*E->parallel.PROCESSOR[lev][m].pass[kkk]=me-((j==1)?1:-1);*/
	  dir = ( (j==1)? 1 : -1);
          E->parallel.PROCESSOR[lev][CPPR].pass[kkk]=E->parallel.loc2proc_map[cap][lx][ly][lz-dir];

          E->parallel.NUM_NODE[lev][CPPR].pass[kkk] = E->parallel.NUM_NNO[lev].bound[ii];

          jj = 0; kf = 0;
          for (kk=1;kk<=E->parallel.NUM_NODE[lev][CPPR].pass[kkk];kk++)   {
            lnode = kk;
            node = E->parallel.NODE[lev][CPPR][lnode].bound[ii];
            E->parallel.EXCHANGE_NODE[lev][CPPR][kk].pass[kkk] = node;
            temp_dims = dims;
                    for(doff=1;doff<=temp_dims;doff++)
                         E->parallel.EXCHANGE_ID[lev][CPPR][++jj].pass[kkk] = E->ID[lev][node].doff[doff];
            }  /* end for node k */

              E->parallel.NUM_NEQ[lev].pass[kkk] = jj;

          }   /* end if */

            }     /* end for j */


      E->parallel.TNUM_PASS[lev] = kkk;



      }        /* end for level */

  if(E->control.verbose) {
    for(lev=E->mesh.gridmax;lev>=E->mesh.gridmin;lev--) {
      fprintf(E->fp_out,"output_communication route surface for lev=%d \n",lev);
    fprintf(E->fp_out,"  me= %d cap=%d pass  %d \n",E->parallel.me,E->sphere.capid[CPPR],E->parallel.TNUM_PASS[lev]);
    for (k=1;k<=E->parallel.TNUM_PASS[lev];k++)   {
      fprintf(E->fp_out,"proc %d and pass  %d to proc %d with %d eqn and %d node\n",E->parallel.me,k,E->parallel.PROCESSOR[lev][CPPR].pass[k],E->parallel.NUM_NEQ[lev].pass[k],E->parallel.NUM_NODE[lev][CPPR].pass[k]);
/*    fprintf(E->fp_out,"Eqn:\n");  */
/*    for (ii=1;ii<=E->parallel.NUM_NEQ[lev][m].pass[k];ii++)  */
/*      fprintf(E->fp_out,"%d %d\n",ii,E->parallel.EXCHANGE_ID[lev][m][ii].pass[k]);  */
/*    fprintf(E->fp_out,"Node:\n");  */
/*    for (ii=1;ii<=E->parallel.NUM_NODE[lev][m].pass[k];ii++)  */
/*      fprintf(E->fp_out,"%d %d\n",ii,E->parallel.EXCHANGE_NODE[lev][m][ii].pass[k]);  */
    }

    }
    fflush(E->fp_out);
  }
}

/* ============================================
 determine communication routs for
 exchange info across the boundaries on the surfaces
 assuming fault nodes are in the top row of processors
 ============================================ */

void regional_parallel_communication_routs_s(E)
  struct All_variables *E;
  {

  int i,ii,j,k,l,node,el,elt,lnode,jj,doff;
  int lev,nno,nox,noz,noy,kkk,kk,kf;
  int p,me,m, nprocz;
  int nprocxl,nprocyl,nproczl;
  void parallel_process_termination();
  FILE *fp1,*fp2;

  char output_file[200];
  const int dims=E->mesh.nsd;

  me = E->parallel.me;
  nprocz = E->parallel.nprocz;

  nprocxl = E->parallel.nprocx;
  nprocyl = E->parallel.nprocy;
  nproczl = E->parallel.nprocz;


        /* determine the communications in horizontal direction        */
  for(lev=E->mesh.gridmax;lev>=E->mesh.gridmin;lev--)       {
    nox = E->lmesh.NOX[lev];
    noz = E->lmesh.NOZ[lev];
    noy = E->lmesh.NOY[lev];
    ii = 0;
    kkk = 0;


        for(i=1;i<=2;i++)       {       /* do YOZ boundaries & OY lines */

        ii ++;
        E->parallel.NUM_PASS[lev].bound[ii] = 1;
        if(E->parallel.me_loc[1]==0 && i==1)
          E->parallel.NUM_PASS[lev].bound[ii] = 0;
        else if(E->parallel.me_loc[1]==nprocxl-1 && i==2)
          E->parallel.NUM_PASS[lev].bound[ii] = 0;

        for (p=1;p<=E->parallel.NUM_PASS[lev].bound[ii];p++)  {
          kkk ++;
              /* determine the pass ID for ii-th boundary and p-th pass */

          E->parallel.sPROCESSOR[lev][CPPR].pass[kkk]=me-((i==1)?1:-1)*nproczl;

              E->parallel.NUM_sNODE[lev][CPPR].pass[kkk] =
                          E->parallel.NUM_NNO[lev].bound[ii]/noz;
          for (k=1;k<=E->parallel.NUM_sNODE[lev][CPPR].pass[kkk];k++)   {
            lnode = k;             /* due to lnode increases in horizontal di first */
            node = (E->parallel.NODE[lev][CPPR][lnode].bound[ii]-1)/noz+1;
            E->parallel.EXCHANGE_sNODE[lev][CPPR][k].pass[kkk] = node;
            }  /* end for node k */

          }   /* end for loop p */
            }  /* end for i */

     ii = 2;
          for(k=1;k<=2;k++)        {      /* do XOZ boundaries & OX lines */

        ii ++;
        E->parallel.NUM_PASS[lev].bound[ii] = 1;
        if(E->parallel.me_loc[2]==0 && k==1)
          E->parallel.NUM_PASS[lev].bound[ii] = 0;
        else if(E->parallel.me_loc[2]==nprocyl-1 && k==2)
          E->parallel.NUM_PASS[lev].bound[ii] = 0;

        for (p=1;p<=E->parallel.NUM_PASS[lev].bound[ii];p++)  {

          kkk ++;
              /* determine the pass ID for ii-th boundary and p-th pass */

          E->parallel.sPROCESSOR[lev][CPPR].pass[kkk]=me-((k==1)?1:-1)*nprocxl*nproczl;

              E->parallel.NUM_sNODE[lev][CPPR].pass[kkk] =
                          E->parallel.NUM_NNO[lev].bound[ii]/noz;

          for (kk=1;kk<=E->parallel.NUM_sNODE[lev][CPPR].pass[kkk];kk++)   {
            lnode = kk;             /* due to lnode increases in horizontal di first */
            node = (E->parallel.NODE[lev][CPPR][lnode].bound[ii]-1)/noz+1;
            E->parallel.EXCHANGE_sNODE[lev][CPPR][kk].pass[kkk] = node;
            }  /* end for node kk */

          }   /* end for loop p */

            }  /* end for k */


    E->parallel.sTNUM_PASS[lev][CPPR] = kkk;



    }   /* end for lev  */
}


/* ================================================
================================================ */

void regional_exchange_id_d(E, U, lev)
 struct All_variables *E;
 double *U;
 int lev;
 {

 int ii,j,jj,m,k;
 double *S[27],*R[27];
 int sizeofk;

 MPI_Status status;

 for (k=1;k<=E->parallel.TNUM_PASS[lev];k++)  {
   sizeofk = (1+E->parallel.NUM_NEQ[lev].pass[k])*sizeof(double);
   S[k]=(double *)malloc( sizeofk );
   R[k]=(double *)malloc( sizeofk );
 }

   for (k=1;k<=E->parallel.TNUM_PASS[lev];k++)  {

     for (j=1;j<=E->parallel.NUM_NEQ[lev].pass[k];j++)
       S[k][j-1] = U[ E->parallel.EXCHANGE_ID[lev][CPPR][j].pass[k] ];

     MPI_Sendrecv(S[k],E->parallel.NUM_NEQ[lev].pass[k],MPI_DOUBLE,
		  E->parallel.PROCESSOR[lev][CPPR].pass[k],1,
		  R[k],E->parallel.NUM_NEQ[lev].pass[k],MPI_DOUBLE,
		  E->parallel.PROCESSOR[lev][CPPR].pass[k],1,
		  E->parallel.world,&status);

     for (j=1;j<=E->parallel.NUM_NEQ[lev].pass[k];j++)
       U[ E->parallel.EXCHANGE_ID[lev][CPPR][j].pass[k] ] += R[k][j-1];

   }           /* for k */

 for (k=1;k<=E->parallel.TNUM_PASS[lev];k++)  {
   free((void*) S[k]);
   free((void*) R[k]);
 }
}


/* ================================================ */
/* ================================================ */
static void exchange_node_d(E, U, lev)
 struct All_variables *E;
 double *U;
 int lev;
 {

 int ii,j,jj,m,k;
 double *S[27],*R[27];
 int sizeofk;

 MPI_Status status;

 for (k=1;k<=E->parallel.TNUM_PASS[lev];k++)  {
   sizeofk = (1+E->parallel.NUM_NODE[lev][CPPR].pass[k])*sizeof(double);
   S[k]=(double *)malloc( sizeofk );
   R[k]=(double *)malloc( sizeofk );
 }   /* end for k */

   for (k=1;k<=E->parallel.TNUM_PASS[lev];k++)   {

     for (j=1;j<=E->parallel.NUM_NODE[lev][CPPR].pass[k];j++)
       S[k][j-1] = U[ E->parallel.EXCHANGE_NODE[lev][CPPR][j].pass[k] ];

     MPI_Sendrecv(S[k],E->parallel.NUM_NODE[lev][CPPR].pass[k],MPI_DOUBLE,
		  E->parallel.PROCESSOR[lev][CPPR].pass[k],1,
		  R[k],E->parallel.NUM_NODE[lev][CPPR].pass[k],MPI_DOUBLE,
		  E->parallel.PROCESSOR[lev][CPPR].pass[k],1,
		  E->parallel.world,&status);

     for (j=1;j<=E->parallel.NUM_NODE[lev][CPPR].pass[k];j++)
       U[ E->parallel.EXCHANGE_NODE[lev][CPPR][j].pass[k] ] += R[k][j-1];
   }

 for (k=1;k<=E->parallel.TNUM_PASS[lev];k++)  {
   free((void*) S[k]);
   free((void*) R[k]);
 }
}

/* ================================================ */
/* ================================================ */

static void exchange_node_f(E, U, lev)
 struct All_variables *E;
 float *U;
 int lev;
{

 int ii,j,jj,m,k;
 float *S[27],*R[27];
 int sizeofk;

 MPI_Status status;

 for (k=1;k<=E->parallel.TNUM_PASS[lev];k++)  {
   sizeofk = (1+E->parallel.NUM_NODE[lev][CPPR].pass[k])*sizeof(float);
   S[k]=(float *)malloc( sizeofk );
   R[k]=(float *)malloc( sizeofk );
 }   /* end for k */


 for (k=1;k<=E->parallel.TNUM_PASS[lev];k++)   {

   for (j=1;j<=E->parallel.NUM_NODE[lev][CPPR].pass[k];j++)
     S[k][j-1] = U[ E->parallel.EXCHANGE_NODE[lev][CPPR][j].pass[k] ];

   MPI_Sendrecv(S[k],E->parallel.NUM_NODE[lev][CPPR].pass[k],MPI_FLOAT,
    E->parallel.PROCESSOR[lev][CPPR].pass[k],1,
    R[k],E->parallel.NUM_NODE[lev][CPPR].pass[k],MPI_FLOAT,
    E->parallel.PROCESSOR[lev][CPPR].pass[k],1,
    E->parallel.world,&status);

   for (j=1;j<=E->parallel.NUM_NODE[lev][CPPR].pass[k];j++)
     U[ E->parallel.EXCHANGE_NODE[lev][CPPR][j].pass[k] ] += R[k][j-1];
 }

 for (k=1;k<=E->parallel.TNUM_PASS[lev];k++)  {
   free((void*) S[k]);
   free((void*) R[k]);
 }
}

/* ================================================ */
/* ================================================ */

void regional_exchange_snode_f(struct All_variables *E, float *U1,
                               float *U2, int lev)
 {

 int ii,j,k,m,kk,t_cap,idb,msginfo[8];
 float *S[27],*R[27];
 int sizeofk;

 MPI_Status status;

 for (k=1;k<=E->parallel.sTNUM_PASS[lev][CPPR];k++)  {
   sizeofk = (1+2*E->parallel.NUM_sNODE[lev][CPPR].pass[k])*sizeof(float);
   S[k]=(float *)malloc( sizeofk );
   R[k]=(float *)malloc( sizeofk );
 }

   for (k=1;k<=E->parallel.sTNUM_PASS[lev][CPPR];k++)  {

     for (j=1;j<=E->parallel.NUM_sNODE[lev][CPPR].pass[k];j++)  {
       S[k][j-1] = U1[ E->parallel.EXCHANGE_sNODE[lev][CPPR][j].pass[k] ];
       S[k][j-1+E->parallel.NUM_sNODE[lev][CPPR].pass[k]]
	 = U2[ E->parallel.EXCHANGE_sNODE[lev][CPPR][j].pass[k] ];
     }

     MPI_Sendrecv(S[k],2*E->parallel.NUM_sNODE[lev][CPPR].pass[k],MPI_FLOAT,
		  E->parallel.sPROCESSOR[lev][CPPR].pass[k],1,
		  R[k],2*E->parallel.NUM_sNODE[lev][CPPR].pass[k],MPI_FLOAT,
		  E->parallel.sPROCESSOR[lev][CPPR].pass[k],1,
		  E->parallel.world,&status);

     for (j=1;j<=E->parallel.NUM_sNODE[lev][CPPR].pass[k];j++)   {
       U1[ E->parallel.EXCHANGE_sNODE[lev][CPPR][j].pass[k] ] += R[k][j-1];
       U2[ E->parallel.EXCHANGE_sNODE[lev][CPPR][j].pass[k] ] +=
	 R[k][j-1+E->parallel.NUM_sNODE[lev][CPPR].pass[k]];
     }

   }

 for (k=1;k<=E->parallel.sTNUM_PASS[lev][CPPR];k++)  {
   free((void*) S[k]);
   free((void*) R[k]);
 }
}
